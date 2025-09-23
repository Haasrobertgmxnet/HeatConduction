<<<<<<< HEAD
﻿from pipeline_2d import pipeline_2d
from pipeline_aryal import pipeline_aryal
from comparison import exec_comparison

# pipeline_aryal()
# pipeline_2d()
exec_comparison()
=======
﻿import numpy as np
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
dt = 0.01

# -------------------------
# PINN Model
# -------------------------
class PINNHeat(nn.Module):
    def __init__(self, hidden_size=50, n_hidden=5):
        super().__init__()
        self.activation= nn.Tanh()
        self.layers = [nn.Linear(3, hidden_size), self.activation]
        for _ in range(n_hidden-1):
            self.layers += [nn.Linear(hidden_size, hidden_size), self.activation]
        self.layers += [nn.Linear(hidden_size, 1)]  # Output: u
        self.model = nn.Sequential(*self.layers)

    def forward(self, x, y, t):
        if x.ndim == 1: x = x.unsqueeze(1)
        if y.ndim == 1: y = y.unsqueeze(1)
        if t.ndim == 1: t = t.unsqueeze(1)
        inputs = torch.cat([x, y, t], dim=1)
        return self.model(inputs)

        output = inputs
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        output = self.layers[-1](output) + 25
        return output

    def fwd(self, xyt):
        return self.model(xyt)

# -------------------------
# Physics Loss using autograd
# -------------------------
def physics_loss(model, x, y, t, alpha, f= 0):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, y, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x , x, grad_outputs=torch.ones_like(u_x),
        create_graph=True)[0]

    u_yy = torch.autograd.grad(u_y , y, grad_outputs=torch.ones_like(u_y),
        create_graph=True)[0]

    residual = u_t - alpha * (u_xx + u_yy) - f
    
    return torch.mean(residual ** 2)

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

xyt = InputData(lx, ly, 1, nx, ny, nt)
x = xyt.get_x()
y = xyt.get_y()
t = xyt.get_t()


# Initial condition t=0
x_ic = xyt.get_x_ic()
y_ic = xyt.get_y_ic()
t_ic = xyt.get_t_ic()

Xyt_ic = torch.stack([x_ic.flatten(), y_ic.flatten(), t_ic.flatten()], dim=1)

# ---- Gaußparameter ----
A = 50.0
x0 = lx / 2
y0 = ly / 2
sigma_x = lx / 10  # Breite in x-Richtung
sigma_y = ly / 10  # Breite in y-Richtung

# 2D-Gaußverteilung auf dem Raster berechnen
u_ic_grid = A * torch.exp(-((x_ic - x0)**2)/(2*sigma_x**2)
                         -((y_ic - y0)**2)/(2*sigma_y**2))

# Als flachen Vektor für PINN / Solver:
u_ic = u_ic_grid.flatten()

Xf = xyt.get_capX().flatten()
Yf = xyt.get_capY().flatten()
# -------------------------
# Masken für jede Randseite
# -------------------------
mask_left   = Xf == 0
mask_right  = Xf == lx
mask_bottom = Yf == 0
mask_top    = Yf == ly

# -------------------------
# Boundary-Punkte extrahieren
# -------------------------
Xyt_left   = xyt.Xyt[Xf == 0]
Xyt_right  = xyt.Xyt[Xf == lx]
Xyt_bottom = xyt.Xyt[Yf == 0]
Xyt_top    = xyt.Xyt[Yf == ly]

# -------------------------
# Normalenvektoren erzeugen
# -------------------------
normals_left   = torch.tensor([[-1.0, 0.0]] * Xyt_left.shape[0], dtype=torch.float32)
normals_right  = torch.tensor([[ 1.0, 0.0]] * Xyt_right.shape[0], dtype=torch.float32)
normals_bottom = torch.tensor([[ 0.0,-1.0]] * Xyt_bottom.shape[0], dtype=torch.float32)
normals_top    = torch.tensor([[ 0.0, 1.0]] * Xyt_top.shape[0], dtype=torch.float32)

# Beispiel: Zugriff auf ersten linken Randpunkt + Normalen
print("\nBeispiel linker Randpunkt:", Xyt_left[0])
print("Seine Normale:", normals_left[0])


# Boundary condition (Dirichlet)
# Take points on boundaries for all t
mask = (Xf == 0) | (Xf == lx) | (Yf == 0) | (Yf == ly)
Xyt_bc = xyt.Xyt[mask]
u_bc = torch.zeros(Xyt_bc.shape[0])
u_bc[:] = 20.0  # boundary temperature

# -------------------------
# Model, optimizer
# -------------------------
model = PINNHeat(hidden_size=50, n_hidden=5)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)

# -------------------------
# Training loop
# -------------------------

epochs0 = 20
epochs = 5*epochs0
lambda_phy = 1.0
lambda_ic = 5.0
lambda_bc = 2.0

# a, b, c = 1.0, 0.0, 20.0

for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss_phy = physics_loss(model, x, y, t, alpha)
    # loss_ic = ic_loss(model, Xyt_ic, u_ic)
    # loss_bc = bc_loss(model, Xyt_bc, u_bc)
    loss_ic = 0
    loss_bc = 0
    
    loss = lambda_phy*loss_phy + lambda_ic*loss_ic + lambda_bc*loss_bc
    loss.backward()
    optimizer.step()
    
    if epoch % epochs0 == 0:
        print(f"Epoch {epoch}, Total Loss={loss.item():.6f}, "
              f"Physics Loss={loss_phy:.6f}, "
              f"Initial Loss={loss_ic:.6f}, "
              f"Boundary Loss={loss_bc:.6f}, "
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
        u_pred = model.fwd(Xyt_vis).reshape(n_vis, n_vis).cpu().numpy()
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
        ax.set_title(f"ML Solution - t = {frame*dt:.4f}")
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
>>>>>>> 3ef016f52a46c7b42cf3629486f2706a79a5bab1
