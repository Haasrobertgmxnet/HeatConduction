import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Physical parameters
# -------------------------
alpha = 0.01
lx, ly = 1.0, 1.0
nt = 50  # number of time steps for training points
nx, ny = 30, 30

# -------------------------
# PINN Model
# -------------------------
class PINNHeat(nn.Module):
    def __init__(self, hidden_size=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh()]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]  # Output: u
        self.model = nn.Sequential(*layers)
    
    def forward(self, xyt):
        return self.model(xyt)

# -------------------------
# Physics Loss using autograd
# -------------------------
def physics_loss(model, xyt, alpha):
    xyt = xyt.clone().requires_grad_(True)
    u = model(xyt)
    
    # Zeitliche Ableitung
    u_t = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0][:, 2:3]
    # Räumliche Ableitungen
    u_x = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    
    u_y = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0][:, 1:2]
    u_yy = torch.autograd.grad(u_y, xyt, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]
    
    loss = ((u_t - alpha*(u_xx + u_yy))**2).mean()
    return loss

# -------------------------
# Initial Condition Loss
# -------------------------
def ic_loss(model, xyt_ic, u_ic):
    u_pred = model(xyt_ic)
    return ((u_pred - u_ic)**2).mean()

# -------------------------
# Boundary Condition Loss (Dirichlet)
# -------------------------
def bc_loss(model, xyt_bc, u_bc):
    u_pred = model(xyt_bc)
    return ((u_pred - u_bc)**2).mean()

def bc_loss_robin(model, xyt_bc, a, b, c, normal):
    """
    Robin boundary condition: a*u + b*du/dn = c
    normal: Tensor shape (N, 2) with (nx, ny) normals for each boundary point
    """
    xyt_bc = xyt_bc.clone().requires_grad_(True)
    u = model(xyt_bc)

    grads = torch.autograd.grad(u, xyt_bc,
                                grad_outputs=torch.ones_like(u),
                                retain_graph=True, create_graph=True)[0]
    grad_x, grad_y = grads[:, 0:1], grads[:, 1:2]
    du_dn = grad_x*normal[:,0:1] + grad_y*normal[:,1:2]

    bc_residual = a*u + b*du_dn - c
    return (bc_residual**2).mean()

def greet():
    import os
    print("=" * 60)
    print(f"Starting script: {os.path.basename(__file__)}")
    print("=" * 60)

greet()

def grid_test(lx, ly, lt, nx, ny, nt):
    x = torch.linspace(0, lx, nx)
    y = torch.linspace(0, ly, ny)
    t = torch.linspace(0, lt, nt)
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    print(f"X : {X}")
    print(f"X[..0]  : {X[:, :, 0]}")
    print(f"Y : {Y}")
    print(f"T : {T}")
    Xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    print(f"Xyt : {Xyt}")

    # Initial condition t=0
    x_ic = X[:, :, 0].flatten()
    y_ic = Y[:, :, 0].flatten()
    t_ic = T[:, :, 0].flatten()

    Xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
    u_ic = torch.zeros_like(x_ic)
    u_ic[nx//2*ny + ny//2] = 50.0  # Hot spot in the center
    print(f"u_ic : {u_ic}")

# grid_test(1, 2, 1, 3, 5, 3)
# grid_test(lx, ly, 1,  nx, ny, nt)

quit()

# -------------------------
# Training Points
# -------------------------
# Interior points
x = torch.linspace(0, lx, nx)
y = torch.linspace(0, ly, ny)
t = torch.linspace(0, 1.0, nt)
X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
Xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

# Initial condition t=0
x_ic = X[:, :, 0].flatten()
y_ic = Y[:, :, 0].flatten()
t_ic = T[:, :, 0].flatten()

Xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
u_ic = torch.zeros_like(x_ic)
u_ic[nx//2*ny + ny//2] = 50.0  # Hot spot in the center

# -------------------------
# Masken für jede Randseite
# -------------------------
mask_left   = (X.flatten() == 0)
mask_right  = (X.flatten() == lx)
mask_bottom = (Y.flatten() == 0)
mask_top    = (Y.flatten() == ly)

# -------------------------
# Boundary-Punkte extrahieren
# -------------------------
Xyt_left   = Xyt[mask_left]
Xyt_right  = Xyt[mask_right]
Xyt_bottom = Xyt[mask_bottom]
Xyt_top    = Xyt[mask_top]

# -------------------------
# Normalenvektoren erzeugen
# -------------------------
normals_left   = torch.tensor([[-1.0, 0.0]] * Xyt_left.shape[0], dtype=torch.float32)
normals_right  = torch.tensor([[ 1.0, 0.0]] * Xyt_right.shape[0], dtype=torch.float32)
normals_bottom = torch.tensor([[ 0.0,-1.0]] * Xyt_bottom.shape[0], dtype=torch.float32)
normals_top    = torch.tensor([[ 0.0, 1.0]] * Xyt_top.shape[0], dtype=torch.float32)

# -------------------------
# Kontrollausgabe
# -------------------------
print("Randpunkte links :", Xyt_left.shape, " Normals:", normals_left.shape)
print("Randpunkte rechts:", Xyt_right.shape, " Normals:", normals_right.shape)
print("Randpunkte unten :", Xyt_bottom.shape, " Normals:", normals_bottom.shape)
print("Randpunkte oben  :", Xyt_top.shape, " Normals:", normals_top.shape)

# Beispiel: Zugriff auf ersten linken Randpunkt + Normalen
print("\nBeispiel linker Randpunkt:", Xyt_left[0])
print("Seine Normale:", normals_left[0])


# Boundary condition (Dirichlet)
# Take points on boundaries for all t
mask = (X.flatten() == 0) | (X.flatten() == lx) | (Y.flatten() == 0) | (Y.flatten() == ly)
Xyt_bc = Xyt[mask]
u_bc = torch.zeros(Xyt_bc.shape[0])
u_bc[:] = 20.0  # boundary temperature

# -------------------------
# Model, optimizer
# -------------------------
model = PINNHeat(hidden_size=64, n_hidden=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# -------------------------
# Training loop
# -------------------------
epochs = 500
lambda_phy = 1.0
lambda_ic = 1.0
lambda_bc = 1.0

a, b, c = 1., 0.0, 20.0

for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss_phy = physics_loss(model, Xyt, alpha)
    loss_ic = ic_loss(model, Xyt_ic, u_ic)
    # loss_bc = bc_loss(model, Xyt_bc, u_bc)

    loss_bc = (
    bc_loss_robin(model, Xyt_left,   a,b,c, normals_left) +
    bc_loss_robin(model, Xyt_right,  a,b,c, normals_right) +
    bc_loss_robin(model, Xyt_bottom, a,b,c, normals_bottom) +
    bc_loss_robin(model, Xyt_top,    a,b,c, normals_top)
    )/4.0
    
    loss = lambda_phy*loss_phy + lambda_ic*loss_ic + lambda_bc*loss_bc
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Total Loss={loss.item():.6f}")

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
plt.imshow(u_ic.reshape(n_vis, n_vis).cpu().numpy(), origin='lower', extent=[0, lx, 0, ly], cmap='hot')
plt.colorbar(label="Temperature")
plt.title("Temperature at start")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

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