import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Grid parameters
# -------------------------
nx, ny = 30, 30
lx, ly = 1.0, 1.0
dx, dy = lx/nx, ly/ny

# -------------------------
# Physical parameters
# -------------------------
alpha = 0.001
dt = 0.0005
nt = 200

# -------------------------
# Initial condition
# -------------------------
center_value = 50.0
u0 = np.zeros((nx, ny))
u0[nx//2, ny//2] = center_value

# replace point initial by gaussian hotspot
sigma = 0.15  # breitengrad; anpassen
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
Xg, Yg = np.meshgrid(x, y, indexing='ij')
xc, yc = lx/2, ly/2
u0 = np.exp(-((Xg-xc)**2 + (Yg-yc)**2) / (2*sigma**2))
u0 = u0 / u0.max() * center_value

# -------------------------
# Boundary condition (Robin)
# -------------------------
a, b, c = 0.0, 1.0, 0.0

# -------------------------
# Device
# -------------------------
device = 'cpu'
if torch.cuda.is_available():
    print("CUDA is available")
    device = 'cuda'
else:
    print("CUDA is NOT available. Using CPU")

# -------------------------
# RNN Model
# -------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x, h_prev, c_prev):
        # x: (B, C, H, W)
        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_h, H, W)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


# -------------------------
# RNN Model
# -------------------------
class HeatRNN(nn.Module):
    def __init__(self, nx, ny, hidden_size=64):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=nx*ny, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, nx*ny)
        
    def forward(self, u_seq):
        # u_seq: (batch, time, nx, ny)
        batch, time, nx, ny = u_seq.shape
        u_flat = u_seq.view(batch, time, nx*ny)  # flatten 2D to 1D
        out, _ = self.rnn(u_flat)
        out = self.fc(out)
        out = out.view(batch, time, nx, ny)      # reshape back to 2D
        return out

# -------------------------
# Boundary condition
# -------------------------
def apply_bc(u, a, b, c, dx, dy):
    u_new = u.clone()
    u_new[:,:,:,0] = (c*dx + b*u[:,:,:,1]) / (a*dx + b)
    u_new[:,:,:, -1] = (c*dx + b*u[:,:,:,-2]) / (a*dx + b)
    u_new[:,:,0,:] = (c*dy + b*u[:,:,1,:]) / (a*dy + b)
    u_new[:,:, -1,:] = (c*dy + b*u[:,:,-2,:]) / (a*dy + b)
    return u_new

# -------------------------
# Generate FD reference sequence
# -------------------------
def generate_sequence(u0, nt, alpha, dt, dx, dy, a, b, c):
    u_seq = []
    u = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    for t in range(nt):
        u_new = u.clone()
        u_new[:,:,1:-1,1:-1] = u[:,:,1:-1,1:-1] + alpha*dt*(
            (u[:,:,2:,1:-1]-2*u[:,:,1:-1,1:-1]+u[:,:,:-2,1:-1])/dx**2 +
            (u[:,:,1:-1,2:]-2*u[:,:,1:-1,1:-1]+u[:,:,1:-1,:-2])/dy**2
        )
        u_new = apply_bc(u_new, a, b, c, dx, dy)
        u_seq.append(u_new)
        u = u_new
    u_seq = torch.cat(u_seq, dim=1)  # shape (1, nt, nx, ny)
    return u_seq

# -------------------------
# Physics Loss using autograd
# -------------------------
def physics_loss(u, alpha, dt, dx, dy):
    u = u.clone().detach().requires_grad_(True)
    batch, nt, nx, ny = u.shape
    # Time derivative
    u_t = (u[:,1:,:,:] - u[:,:-1,:,:]) / dt

    # Spatial derivatives via finite differences (still differentiable)
    u_xx = (u[:,:,2:,1:-1] - 2*u[:,:,1:-1,1:-1] + u[:,:,:-2,1:-1]) / dx**2
    u_yy = (u[:,:,1:-1,2:] - 2*u[:,:,1:-1,1:-1] + u[:,:,1:-1,:-2]) / dy**2

    u_xx_yy = u_xx[:, :-1, :, :] + u_yy[:, :-1, :, :]
    u_t_interior = u_t[:, :, 1:-1, 1:-1]

    loss = ((u_t_interior - alpha*u_xx_yy)**2).mean()
    return loss

# -------------------------
# Boundary Loss
# -------------------------
def boundary_loss(u, a, b, c, dx, dy):
    du_left = (u[:,:,:,1] - u[:,:,:,0]) / dx
    du_right = (u[:,:,:,-1] - u[:,:,:,-2]) / dx
    du_bottom = (u[:,:,1,:] - u[:,:,0,:]) / dy
    du_top = (u[:,:,-1,:] - u[:,:,-2,:]) / dy
    loss = ((a*u[:,:,:,0] + b*du_left - c)**2).mean()
    loss += ((a*u[:,:,:,-1] + b*du_right - c)**2).mean()
    loss += ((a*u[:,:,0,:] + b*du_bottom - c)**2).mean()
    loss += ((a*u[:,:,-1,:] + b*du_top - c)**2).mean()
    return loss

def energy_loss_(u):
    """
    u: (batch, time, nx, ny)
    Berechnet den Energie-/Massenverlust
    """
    # Gesamtenergie pro Zeitschritt:
    total_energy = u.sum(dim=(2,3))   # shape (batch, time)
    # Erste Zeitschrittenergie als Referenz:
    ref_energy = total_energy[:,0:1]  # shape (batch,1)
    # Differenz zu Referenz (Broadcasting):
    diff = total_energy - ref_energy
    loss = (diff**2).mean()
    return loss

def energy_loss(u):
    """
    Energy difference between consecutive timesteps
    """
    # (batch,time,nx,ny)
    total_energy = u.sum(dim=(2,3))  # (batch,time)
    diff = total_energy[:,1:] - total_energy[:,:-1]
    return (diff**2).mean()

def greet():
    import os
    print("=" * 60)
    print(f"Starting script: {os.path.basename(__file__)}")
    print("=" * 60)

greet()

# -------------------------
# Training
# -------------------------
u_seq = generate_sequence(u0, nt, alpha, dt, dx, dy, a, b, c).to(device)
model = HeatRNN(nx, ny).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 regularization

loss_fn = nn.MSELoss()
epochs = 200
lambda_mse, lambda_phy, lambda_bc, lambda_energy = 1.0, 0.5, 0.5, 20.0
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(u_seq[:,:-1,:,:])
    

    mse_loss = loss_fn(output, u_seq[:,1:,:,:])
    phy_loss = physics_loss(output, alpha, dt, dx, dy)
    bc_loss = boundary_loss(output, a, b, c, dx, dy)
    en_loss  = energy_loss(output)
    loss = lambda_mse * mse_loss + lambda_phy * phy_loss + lambda_bc*bc_loss + lambda_energy*en_loss

    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print("=" * 60)
        print(f"Epoch {epoch}, Loss={loss.item():.6f}")
        print(f"MSE Loss={mse_loss:.6f}")
        print(f"Physics Loss={phy_loss:.6f}")
        print(f"Boundary Loss={bc_loss:.6f}")
        print(f"Energy Loss={en_loss:.6f}")
# -------------------------
# Prediction
# -------------------------
u_pred = [u_seq[:,0:1,:,:]]
for t in range(nt-1):
    input_seq = torch.cat(u_pred, dim=1)
    out = model(input_seq[:,-1:,:,:])
    out = apply_bc(out, a, b, c, dx, dy)
    u_pred.append(out)
u_pred = torch.cat(u_pred, dim=1)
print("Prediction completed!")

# u_pred: (1, nt, nx, ny) → wir nehmen die erste Batch
u_vis = u_pred[0].detach().cpu().numpy()  # shape: (nt, nx, ny)

# -------------------------
# Abfragepunkte fürs Visualisieren
# -------------------------
n_vis = 30  # Auflösung
x_vis = torch.linspace(0, lx, n_vis)
y_vis = torch.linspace(0, ly, n_vis)
t_vis = torch.linspace(0, 1.0, 100)  # feiner zeitlicher Verlauf

# -------------------------
# Animation or slider
# -------------------------
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

results = u_vis

def anim_slide():
    # Annahme: 'results' ist aus deinem Crank-Nicolson-Solver:
    # results.shape = (nt+1, nx, ny)
    nt = results.shape[0]
    lx, ly = 1.0, 1.0
    dt = 0.0005  # Zeitschritt (musst du ggf. anpassen)

    # -------------------------------
    # Figure und Achsen erzeugen
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6,5))
    plt.subplots_adjust(bottom=0.25)  # Platz für Slider und Button

    cax = ax.imshow(results[0], origin='lower', extent=[0, lx, 0, ly],
                    cmap='hot', vmin=results.min(), vmax=results.max())
    fig.colorbar(cax, label="Temperature")

    ax.set_title("t = 0.0000")

    # Slider-Achse
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Time", 0, nt-1, valinit=0, valstep=1)

    # Play/Stop-Button-Achse
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Play', color='lightgray', hovercolor='0.85')

    # -------------------------------
    # Interaktive Logik
    # -------------------------------
    playing = False  # Zustand

    def update_slider(val):
        """Update-Funktion fuer Slider"""
        frame = int(slider.val)
        cax.set_data(results[frame])
        ax.set_title(f"t = {frame*dt:.4f}")
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    def play_animation(event):
        """Play/Stop-Button gedrueckt"""
        nonlocal playing
        playing = not playing
        if playing:
            button.label.set_text('Stop')
            fig.canvas.start_event_loop(0.001)  # kurz abwarten
            run_animation()
        else:
            button.label.set_text('Play')

    button.on_clicked(play_animation)

    def run_animation():
        """Automatisch durchlaufen"""
        nonlocal playing
        # wichtig: playing überprüfen in jeder Iteration
        for frame in range(int(slider.val)+1, nt):
            if not playing:
                break
            slider.set_val(frame)   # das triggert update_slider automatisch
            plt.pause(0.05)         # Geschwindigkeit anpassen
        button.label.set_text('Play')
        # Wenn am Ende angekommen:
        # global playing
        playing = False

    plt.show()

anim_slide()