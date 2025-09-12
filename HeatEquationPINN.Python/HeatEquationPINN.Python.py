import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# -------------------------
# Physical parameters
# -------------------------
alpha = 0.01
lx, ly = 1.0, 1.0
nt = 50  # number of time steps for training points
nx, ny = 30, 30
dt = 0.01

class InputData:
    def __init__(self, lx, ly, lt, nx, ny, nt):
        x = torch.linspace(0, lx, nx)
        y = torch.linspace(0, ly, ny)
        t = torch.linspace(0, lt, nt)
        self.X, self.Y, self.T = torch.meshgrid(x, y, t, indexing='ij')
        self.Xyt = torch.stack([self.X.flatten(), self.Y.flatten(), self.T.flatten()], dim=1)

    def get_data(self):
        return self.Xyt

    def get_tuple(self, j):
        return self.Xyt[j]

    def get_capX(self):
        return self.X

    def get_capY(self):
        return self.Y

    def get_capT(self):
        return self.T

    def get_x_ic(self):
        return self.X[:, :, 0]

    def get_y_ic(self):
        return self.Y[:, :, 0]

    def get_t_ic(self):
        return self.T[:, :, 0]

    def get_x(self):
        return self.Xyt[:,0]

    def get_y(self):
        return self.Xyt[:,1]

    def get_t(self):
        return self.Xyt[:,2]

    def print_x(self):
        torch.set_printoptions(threshold=torch.inf)
        print(f"x: {xyt.get_x()}")
        torch.set_printoptions(profile="default")

    def print_y(self):
        torch.set_printoptions(threshold=torch.inf)
        print(f"y: {xyt.get_y()}")
        torch.set_printoptions(profile="default")

    def print_t(self):
        torch.set_printoptions(threshold=torch.inf)
        print(f"t: {xyt.get_t()}")
        torch.set_printoptions(profile="default")

# -------------------------
# PINN Model
# -------------------------
class PINNHeat(nn.Module):
    def __init__(self, hidden_size=50, n_hidden=5, activation= nn.Tanh()):
        super().__init__()
        self.layers = [nn.Linear(3, hidden_size), activation]
        for _ in range(n_hidden-1):
            self.layers += [nn.Linear(hidden_size, hidden_size), activation]
        self.layers += [nn.Linear(hidden_size, 1)]  # Output: u
        self.model = nn.Sequential(*self.layers)

    def forward(self, xyt):
        return self.model(xyt)

def generate_boundary_points_and_normals(x_min, x_max, y_min, y_max, t_min, t_max, n_per_side=50):
    # t als Zufallswerte für Randpunkte
    t_rand = torch.rand(n_per_side,1)*(t_max - t_min) + t_min

    # Linke Seite x=x_min
    y_left = torch.rand(n_per_side,1)*(y_max - y_min) + y_min
    x_left = torch.full_like(y_left, x_min)
    normals_left = torch.tensor([[-1.0,0.0,0.0]]).repeat(n_per_side,1)

    # Rechte Seite x=x_max
    y_right = torch.rand(n_per_side,1)*(y_max - y_min) + y_min
    x_right = torch.full_like(y_right, x_max)
    normals_right = torch.tensor([[+1.0,0.0,0.0]]).repeat(n_per_side,1)

    # Untere Seite y=y_min
    x_bottom = torch.rand(n_per_side,1)*(x_max - x_min) + x_min
    y_bottom = torch.full_like(x_bottom, y_min)
    normals_bottom = torch.tensor([[0.0,-1.0,0.0]]).repeat(n_per_side,1)

    # Obere Seite y=y_max
    x_top = torch.rand(n_per_side,1)*(x_max - x_min) + x_min
    y_top = torch.full_like(x_top, y_max)
    normals_top = torch.tensor([[0.0,+1.0,0.0]]).repeat(n_per_side,1)

    # Alle Punkte zusammensetzen
    xyt_left   = torch.cat([x_left, y_left, t_rand], dim=1)
    xyt_right  = torch.cat([x_right, y_right, t_rand], dim=1)
    xyt_bottom = torch.cat([x_bottom, y_bottom, t_rand], dim=1)
    xyt_top    = torch.cat([x_top, y_top, t_rand], dim=1)

    xyt_boundary = torch.cat([xyt_left, xyt_right, xyt_bottom, xyt_top], dim=0)
    normals_boundary = torch.cat([normals_left, normals_right, normals_bottom, normals_top], dim=0)

    return xyt_boundary, normals_boundary

# def initial_loss(model, x, y, t, alpha, f= 0):
# -------------------------
# Physics Loss using autograd
# -------------------------
def physics_loss(model, xyt, alpha, f=0.0):
    # Eingabe differenzierbar machen
    xyt = xyt.clone().detach().requires_grad_(True)

    # Vorwärtsdurchlauf
    u = model(xyt)  # shape [N,1] oder [N]

    # Zeitableitung
    grads = torch.autograd.grad(
        outputs=u,
        inputs=xyt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]  # shape [N,3]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    # Zweite Ableitungen (xx und yy)
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=xyt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        outputs=u_y,
        inputs=xyt,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
    )[0][:, 1:2]

    # Residuum
    residual = u_t - alpha * (u_xx + u_yy) - f

    # Mittlerer quadratischer Fehler
    return torch.mean(residual ** 2)

def boundary_loss_robin(model, xyt_boundary, normal_vectors, a=None, b=None, c=None):
    # Neumann-Rand-Loss
    xyt_boundary = xyt_boundary.clone().detach().requires_grad_(True)
    u_boundary = model(xyt_boundary)
    grads_boundary = torch.autograd.grad(u_boundary, xyt_boundary,
                                         torch.ones_like(u_boundary),
                                         create_graph=True)[0]
    # Skalarprodukt Gradient mit Normalenrichtung
    du_dn = torch.sum(grads_boundary * normal_vectors, dim=1, keepdim=True)

    if a is None:
        return 0.0
    if b is None:
        return 0.0
    if c is None:
        return 0.0
    return torch.mean((b*du_dn - c*u_boundary)**2)

def boundary_loss_dirichlet(model, xyt_boundary, u_boundary=None):
    # Dirichlet-Rand-Loss
    if u_boundary is not None:
        u_bc = model(xyt_boundary)
        return torch.mean((u_bc - u_boundary)**2)
    else:
        return 0.0

def boundary_loss_neumann(model, xyt_boundary, normal_vectors, du_dn_boundary=None):
    # Neumann-Rand-Loss
    xyt_boundary = xyt_boundary.clone().detach().requires_grad_(True)
    u_boundary = model(xyt_boundary)
    grads_boundary = torch.autograd.grad(u_boundary, xyt_boundary,
                                         torch.ones_like(u_boundary),
                                         create_graph=True)[0]
    # Skalarprodukt Gradient mit Normalenrichtung
    du_dn = torch.sum(grads_boundary * normal_vectors, dim=1, keepdim=True)

    if du_dn_boundary is not None:
        return torch.mean((du_dn - du_dn_boundary)**2)
    else:
        return 0.0

def sample_interior_points(xyt, n_samples=1024):
    idx = torch.randperm(xyt.shape[0])[:n_samples]
    return xyt[idx]

xyt = InputData(lx, ly, 1, nx, ny, nt)
x = xyt.get_x()
y = xyt.get_y()
t = xyt.get_t()


# Initial condition t=0
x_ic = xyt.get_x_ic()
y_ic = xyt.get_y_ic()
t_ic = xyt.get_t_ic()

Xyt_ic = torch.stack([x_ic.flatten(), y_ic.flatten(), t_ic.flatten()], dim=1)

def gaussian2D(xyt, scal=1.0):
    # Nur x und y Spalten nehmen (erste zwei)
    XY = xyt[:, :2]  # Shape [N,2]

    sigma = 1.0
    # Gauss: exp(- (x² + y²) / sigma)
    result = torch.exp(-((XY[:, 0])**2 + (XY[:, 1])**2) / sigma)

    return scal * result

u_ic = gaussian2D(Xyt_ic)

xyt_boundary, normals = generate_boundary_points_and_normals(
    x_min=0.0, x_max=lx, y_min=0.0, y_max=ly, t_min=0.0, t_max=1.0, n_per_side=50
)

def time_capsule(quiet, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    time_diff = time.time() - start
    if quiet:
        return result
    print(f"{func.__qualname__} executed in {time_diff:.6f} seconds")
    return result

def greet():
    import os
    print("=" * 60)
    print(f"Starting script: {os.path.basename(__file__)}")
    print("=" * 60)

greet()

# -------------------------
# Model, optimizer
# -------------------------
model = PINNHeat(hidden_size=64, n_hidden=5)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# -------------------------
# Training loop
# -------------------------

epochs0 = 20
epochs = 10*epochs0
lambda_phy = 1.0
lambda_ic = 100.0   # IC oft relativ stark gewichten
lambda_bc = 100.0   # BC ebenfalls stark gewichten

a, b, c = 1.0, 0.0, 20.0

for epoch in range(epochs):
    # print("=" * 50)
    # print(f"Epoch : {epoch}")
    start = time.time()
    time_capsule(True, optimizer.zero_grad)
    xyt_batch = sample_interior_points(xyt.Xyt, 4096) # 16384
    # loss_phy = time_capsule(True, physics_loss,model, xyt_batch, alpha, gaussian2D(xyt_batch))
    # loss_bc = boundary_loss_robin(model, xyt_boundary, normals, 1.0, 0.0, c*torch.ones(xyt_boundary.shape[0],1))
    loss_phy = time_capsule(True, physics_loss,model, xyt_batch, alpha)
    loss_bc = boundary_loss_dirichlet(model, xyt_boundary, 0.0*c*torch.ones(xyt_boundary.shape[0],1))
    # loss_phy = time_capsule(physics_loss,model, xyt.Xyt, alpha, gaussian2D(xyt.Xyt))
    u_pred_ic = model(Xyt_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    # loss_bc = 0
    loss = lambda_phy*loss_phy + lambda_ic*loss_ic + lambda_bc*loss_bc

    time_capsule(True, loss.backward)
    time_capsule(True, optimizer.step)
    time_diff = time.time() -start
    
    if epoch % epochs0 == 0:
        print(f"Epoch {epoch}, Total Loss={loss.item():.6f}, "
              f"Physics Loss={loss_phy:.6f}, "
              f"Initial Loss={loss_ic:.6f}, "
              f"Boundary Loss={loss_bc:.6f}, "
              f"Time={time_diff:.4f}, "
              )

print("Training completed!")

# -------------------------
# Prediction
# -------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Abfragepunkte fürs Visualisieren
# -------------------------
n_vis = 30  # Auflösung
x_vis = torch.linspace(0, lx, n_vis)
y_vis = torch.linspace(0, ly, n_vis)
t_vis = torch.linspace(0, 1.0, 100)  # feiner zeitlicher Verlauf
Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')

u_frames = []
with torch.no_grad():
    for tval in t_vis:
        Xyt_vis = torch.stack([
            Xv.flatten(),
            Yv.flatten(),
            torch.full_like(Xv.flatten(), tval)
        ], dim=1)
        u_pred = model(Xyt_vis).reshape(n_vis, n_vis).cpu().numpy()
        u_frames.append(u_pred)

u_frames = np.array(u_frames)  # shape (nt, nx, ny)

# -------------------------
# Einzelbild (z.B. t = 0)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_ic.reshape(n_vis, n_vis).cpu().numpy(), origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')
plt.colorbar(label="Temperature")
plt.title("Temperature at start")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Einzelbild (z.B. erstes t)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_frames[0], origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')
plt.colorbar(label="Temperature")
plt.title("Temperature at first timestep")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Einzelbild (z.B. zweites t)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_frames[1], origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')
plt.colorbar(label="Temperature")
plt.title("Temperature at 2nd timestep")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Einzelbild (z.B. letztes t)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_frames[-1], origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')
plt.colorbar(label="Temperature")
plt.title("Temperature at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


from matplotlib.widgets import Slider, Button

# Slider animation function für ML-Lösung
def anim_slide():
    results = u_frames
    nt_vis, nx, ny = results.shape
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(bottom=0.25)

    vmin, vmax = results.min(), results.max()
    cax = ax.imshow(results[0], origin='lower', extent=[0, lx, 0, ly],
                    cmap='coolwarm', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, label="Temperature")
    ax.set_title(f"ML Solution - t = 0.0000")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Time", 0, nt_vis-1, valinit=0, valstep=1)

    # Play/Stop button
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
    button = Button(ax_button, 'Play', color='lightgray', hovercolor='0.85')

    playing = False

    def update_slider(val):
        frame = int(slider.val)
        cax.set_data(results[frame])
        ax.set_title(f"Frame = {frame}")
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
        for frame in range(int(slider.val)+1, nt_vis):
            if not playing:
                break
            slider.set_val(frame)
            plt.pause(0.05)
        button.label.set_text('Play')
        playing = False

    plt.show()

# Run slider animation
anim_slide()

quit()

# -------------------------
# Einzelbild (z.B. letztes t)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_frames[-1], origin='lower', extent=[0, lx, 0, ly], cmap='hot')
plt.colorbar(label="Temperature")
plt.title("Temperature at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Animation
# -------------------------
fig, ax = plt.subplots(figsize=(6,5))
cax = ax.imshow(u_frames[0], origin='lower', extent=[0, lx, 0, ly],
                cmap='hot', vmin=u_frames.min(), vmax=u_frames.max())
fig.colorbar(cax, label="Temperature")

def update(frame):
    cax.set_data(u_frames[frame])
    ax.set_title(f"t = {t_vis[frame]:.2f}")
    return cax,

ani = animation.FuncAnimation(fig, update, frames=len(t_vis), interval=100)
plt.show()

from matplotlib.widgets import Slider

# Visualisiere die Lösung als Funktion der Zeit
n_vis = 30
x_vis = torch.linspace(0, lx, n_vis)
y_vis = torch.linspace(0, ly, n_vis)
Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')

t_vis = torch.linspace(0, 1.0, 100)
u_frames = []
with torch.no_grad():
    for tval in t_vis:
        Xyt_vis = torch.stack([
            Xv.flatten(),
            Yv.flatten(),
            torch.full_like(Xv.flatten(), tval)
        ], dim=1)
        u_pred = model(Xyt_vis).reshape(n_vis, n_vis).cpu().numpy()
        u_frames.append(u_pred)

u_frames = np.array(u_frames)

# -------------------------
# Slider-Plot
# -------------------------
fig, ax = plt.subplots(figsize=(6,5))
plt.subplots_adjust(bottom=0.2)

cax = ax.imshow(u_frames[0], origin='lower', extent=[0, lx, 0, ly],
                cmap='hot', vmin=u_frames.min(), vmax=u_frames.max())
fig.colorbar(cax, label="Temperature")
ax.set_title(f"t = {t_vis[0]:.2f}")

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, "Time", 0, len(t_vis)-1, valinit=0, valstep=1)

def update(val):
    frame = int(slider.val)
    cax.set_data(u_frames[frame])
    ax.set_title(f"t = {t_vis[frame]:.2f}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()