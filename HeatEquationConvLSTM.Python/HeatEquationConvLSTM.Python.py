import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -------------------------
# Grid & Physics
# -------------------------
nx, ny = 30, 30
lx, ly = 1.0, 1.0
dx, dy = lx/nx, ly/ny
alpha = 1.438e-7
dt = 0.05
nt = 500

def solve_heat_equation(u0, alpha, dt, dx, dy, nt, a=0, b=1, c=0):
    u = u0.copy()
    H, W = u.shape
    u_seq = np.zeros((nt, H, W), dtype=np.float32)
    u_seq[0] = u

    for n in range(1, nt):
        # Innenpunkte (ohne Rand)
        u_xx = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / dx**2
        u_yy = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / dy**2

        u_new = u.copy()
        u_new[1:-1,1:-1] = u[1:-1,1:-1] + alpha*dt*(u_xx + u_yy)

        # Robin-Randbedingungen anwenden
        u_new = apply_bc_numpy(u_new, a,b,c,dx,dy)

        u = u_new
        u_seq[n] = u
    return u_seq

def apply_bc_numpy(u, a,b,c,dx,dy):
    """NumPy-Version der Randbedingungen"""
    u = u.copy()
    # linke/rechte Ränder
    u[:,0]  = (c*dx + b*u[:,1]) / (a*dx + b + 1e-8)
    u[:,-1] = (c*dx + b*u[:,-2]) / (a*dx + b + 1e-8)
    # untere/obere Ränder
    u[0,:]  = (c*dy + b*u[1,:]) / (a*dy + b + 1e-8)
    u[-1,:] = (c*dy + b*u[-2,:]) / (a*dy + b + 1e-8)
    return u


def gen_initial_data1(peak_value = 50.0):
    # Initial condition (hot spot)
    u0 = np.zeros((nx, ny), dtype=np.float32)
    u0[nx//2-1, ny//2] = 0.5*peak_value
    u0[nx//2+1, ny//2] = 0.5*peak_value
    u0[nx//2, ny//2-1] = 0.5*peak_value
    u0[nx//2, ny//2+1] = 0.5*peak_value
    u0[nx//2, ny//2] = peak_value
    return u0


def gen_initial_data2(peak_value = 50.0):
    # Initial condition (blurred hot spot)
    sigma = 0.15  # std dev
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    xc, yc = lx/2, ly/2
    u0 = np.exp(-((Xg-xc)**2 + (Yg-yc)**2) / (2*sigma**2))
    u0 = u0 / u0.max() * peak_value
    return np.float32(u0)

u0 = gen_initial_data2(50.)

# Robin BC parameters
a, b, c = 0,1,0  # homogeneous Neumann as example
u_seq_np = solve_heat_equation(u0, alpha, dt, dx, dy, nt, a,b,c)  # (T,H,W)

# Visualisierung
plt.figure(figsize=(6,5))
plt.imshow(u_seq_np[0], origin='lower', extent=[0,lx,0,ly], cmap='coolwarm')
plt.colorbar(label='Temperature')
plt.title('Explicit Euler')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# ConvLSTM Cell
# -------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4*hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i,f,o,g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c_prev + i*g
        h = o*torch.tanh(c)
        return h, c

# -------------------------
# ConvLSTM Module
# -------------------------
class HeatConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_seq):
        # x_seq: (B, T, C=1, H, W)
        B, T, C, H, W = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        outputs = []
        for t in range(T):
            x_t = x_seq[:,t]
            h, c = self.cell(x_t, h, c)
            out = self.output_conv(h)
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B,T,1,H,W)

# -------------------------
# Boundary conditions
# -------------------------
def apply_bc(u, a, b, c, dx, dy):
    """
    u: (B,T,C,H,W) oder (B,C,H,W)
    """
    dims = u.dim()
    if dims == 5:
        u_new = u.clone()
        # linke/rechte Ränder
        u_new[:,:,:,:,0]  = (c*dx + b*u_new[:,:,:,:,1]) / (a*dx + b + 1e-8)
        u_new[:,:,:,:,-1] = (c*dx + b*u_new[:,:,:,:,-2]) / (a*dx + b + 1e-8)
        # untere/obere Ränder
        u_new[:,:,:,0,:]  = (c*dy + b*u_new[:,:,:,1,:]) / (a*dy + b + 1e-8)
        u_new[:,:,:, -1,:] = (c*dy + b*u_new[:,:,:, -2,:]) / (a*dy + b + 1e-8)
        return u_new
    elif dims == 4:
        u_new = u.clone()
        u_new[:,:,:,0]  = (c*dx + b*u_new[:,:,:,1]) / (a*dx + b + 1e-8)
        u_new[:,:,:,-1] = (c*dx + b*u_new[:,:,:,-2]) / (a*dx + b + 1e-8)
        u_new[:,:,0,:]  = (c*dy + b*u_new[:,:,1,:]) / (a*dy + b + 1e-8)
        u_new[:,:,-1,:] = (c*dy + b*u_new[:,:,-2,:]) / (a*dy + b + 1e-8)
        return u_new
    else:
        raise ValueError(f"Unexpected tensor shape {u.shape}")

# -------------------------
# Physics loss
# -------------------------
def physics_loss(u, alpha, dt, dx, dy):
    """
    u: (B,T,C,H,W)
    """
    u_t = (u[:,1:,:,:,:] - u[:,:-1,:,:,:]) / dt  # (B,T-1,C,H,W)
    
    # Innenpunkte (H-2, W-2)
    u_interior = u[:,:-1,:,1:-1,1:-1]  # (B,T-1,C,H-2,W-2)
    
    # x-Richtung
    u_xx = (u[:,:-1,:,1:-1,2:] - 2*u_interior + u[:,:-1,:,1:-1,:-2]) / dx**2
    # y-Richtung
    u_yy = (u[:,:-1,:,2:,1:-1] - 2*u_interior + u[:,:-1,:, :-2,1:-1]) / dy**2
    
    u_lap = u_xx + u_yy
    
    u_t_interior = u_t[:,:,:,1:-1,1:-1]
    loss = ((u_t_interior - alpha * u_lap)**2).mean()
    return loss

# -------------------------
# Boundary loss
# -------------------------
def boundary_loss(u, a, b, c, dx, dy):
    u_bc = apply_bc(u, a, b, c, dx, dy)
    return ((u - u_bc)**2).mean()

def greet():
    import os
    print("=" * 60)
    print(f"Starting script: {os.path.basename(__file__)}")
    print("=" * 60)

greet()

# -------------------------
# Training
# -------------------------
u_seq = torch.tensor(u0, device=device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
# u_seq = torch.tensor(u_seq_np, device=device).unsqueeze(0).unsqueeze(2)  # (B,T,C,H,W)
u_seq = u_seq.expand(1, nt, 1, *u0.shape).contiguous() # u_seq = u_seq.repeat(1, nt, 1, 1, 1)

# print(f" Is u_seq needed? {u_seq.is_contiguous()}")  # True/False

model = HeatConvLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()
    # output = model(u_seq[:,:-1])  # vielleicht nur T-1 Schritte rein oder output = model(u_seq)
    output = model(u_seq)
    output = apply_bc(output, a, b, c, dx, dy)

    # mse_loss = ((output - u_seq)**2).mean()
    mse_loss = ((output[:,0] - u_seq[:,0])**2).mean() 
    phy_loss = physics_loss(output, alpha, dt, dx, dy)
    bc_loss  = boundary_loss(output, a, b, c, dx, dy)
    loss = 0*mse_loss + 100*phy_loss + 0.1*bc_loss
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.6f}, MSE={mse_loss.item():.6f}, "
              f"Physics={phy_loss.item():.6f}, BC={bc_loss.item():.6f}, "
              f"Max={output.max().item():.4f}, "
              f"Min={output.min().item():.4f}")

# -------------------------
# Visualization / Slider Animation
# -------------------------
with torch.no_grad():
    out = model(u_seq)
    out = apply_bc(out, a, b, c, dx, dy)

u_vis = out[0,:,0].cpu().numpy()  # shape (T,H,W)

# Slider animation function
def anim_slide():
    results = u_vis
    nt, nx, ny = results.shape
    fig, ax = plt.subplots(figsize=(6,5))
    plt.subplots_adjust(bottom=0.25)

    cax = ax.imshow(results[0], origin='lower', extent=[0, lx, 0, ly],
                    cmap='coolwarm', vmin=results.min(), vmax=results.max())
    fig.colorbar(cax, label="Temperature")
    ax.set_title("t = 0.0000")

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Time", 0, nt-1, valinit=0, valstep=1)

    # Play/Stop button
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Play', color='lightgray', hovercolor='0.85')

    playing = False

    def update_slider(val):
        frame = int(slider.val)
        cax.set_data(results[frame])
        ax.set_title(f"t = {frame*dt:.4f}")
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    def play_animation(event):
        nonlocal playing
        playing = not playing
        if playing:
            button.label.set_text('Stop')
            run_animation()
        else:
            button.label.set_text('Play')

    button.on_clicked(play_animation)

    def run_animation():
        nonlocal playing
        for frame in range(int(slider.val)+1, nt):
            if not playing:
                break
            slider.set_val(frame)
            plt.pause(0.05)
        button.label.set_text('Play')
        playing = False

    plt.show()

# Run slider animation
anim_slide()