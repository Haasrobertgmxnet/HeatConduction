import time
from token import STAR
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# Parameter
# -------------------------
alpha = 1e-4 # 1.438e-7 # 0.001
lx, ly = 1.0, 1.0
nt = 100
nx, ny = 30, 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# InputData (wie vorher, nur sicherstellen, dass Xyt float)
# -------------------------
class InputData:
    def __init__(self, lx, ly, lt, nx, ny, nt):
        x = torch.linspace(0, lx, nx, dtype=torch.float32)
        y = torch.linspace(0, ly, ny, dtype=torch.float32)
        t = torch.linspace(0, lt, nt, dtype=torch.float32)
        self.X, self.Y, self.T = torch.meshgrid(x, y, t, indexing='ij')
        self.Xyt = torch.stack([self.X.flatten(), self.Y.flatten(), self.T.flatten()], dim=1).float()
        self.Xyt.requires_grad_(True)

    def get_xyt(self):
        return self.Xyt

    def get_x_ic(self):
        return self.X[:, :, 0]

    def get_y_ic(self):
        return self.Y[:, :, 0]

    def get_t_ic(self):
        return self.T[:, :, 0]

xyt = InputData(lx, ly, 1.0, nx, ny, nt)
# gesamte Punktwolke (N,3)
XYT_all = xyt.get_xyt().to(device)

# -------------------------
# Modell (wie vorher)
# -------------------------
class PINNHeat(nn.Module):
    def __init__(self, hidden_size=50, n_hidden=5, activation=nn.Tanh()):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), activation]
        for _ in range(n_hidden-1):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers += [nn.Linear(hidden_size, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, xyt):
        return 25.0 + self.model(xyt)

model = PINNHeat(hidden_size=50, n_hidden=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# -------------------------
# Hilfsfunktionen
# -------------------------
def gaussian2D(xyt, lx=1.0, ly=1.0, sigma=0.15):
    # xyt: (N,3) oder (N,2) -> benutzt nur die ersten 2 Spalten
    XY = xyt[:, :2]
    cx, cy = lx/2.0, ly/2.0
    r2 = (XY[:,0]-cx)**2 + (XY[:,1]-cy)**2
    return torch.exp(-r2 / (2.0 * sigma**2)).unsqueeze(1)
    # Normiere auf 1 (optional, aber konsistent)
    g = g / (g.max().detach() + 1e-12)
    return g

def initial_func1(xyt, scal=1.0, lx=1.0, ly=1.0, sigma=0.15):
    print(f"initial_func, sigma : {sigma:.6}")
    return scal * gaussian2D(xyt, lx, ly, sigma)

def initial_func2(xyt, const_val=25.0):
    return torch.full_like(xyt[:,0], const_val)

def initial_func(xyt):
    return initial_func2(xyt)

def heat_source(xyt):
    return 5000.0*gaussian2D(xyt, 1.0, 1.0, 0.1)

def sample_interior_points(xyt, n_samples=1024):
    n_total = xyt.shape[0]
    n = min(n_samples, n_total)
    idx = torch.randperm(n_total)[:n]
    return xyt[idx]

# -------------------------
# 2) Biased Sampler: mehr Punkte in kleinem Zeitfenster nahe t=0
# -------------------------
import numpy as np
def sample_interior_points_biased(xyt_all, n_samples=1024, frac_near0=0.3, t_eps=0.02):
    """
    Wähle frac_near0-Anteil der Samples mit t <= t_eps (falls verfügbar),
    den Rest zufällig aus der ganzen Domäne.
    """
    n_total = xyt_all.shape[0]
    n_near = int(n_samples * frac_near0)
    # Indices mit t <= t_eps
    mask_near = (xyt_all[:,2] <= t_eps).cpu().numpy()
    idx_near_all = np.where(mask_near)[0]
    selected = []

    if len(idx_near_all) > 0:
        take = min(n_near, len(idx_near_all))
        perm = np.random.permutation(len(idx_near_all))[:take]
        selected.extend(idx_near_all[perm].tolist())

    # rest zufällig
    n_rest = n_samples - len(selected)
    rem = np.random.permutation(n_total)[:n_rest].tolist()
    selected.extend(rem)

    selected = torch.tensor(selected, dtype=torch.long, device=xyt_all.device)
    return xyt_all[selected]


# physics loss: akzeptiert f=None oder f shape (N,1)
def physics_loss(model, xyt, alpha, f=None):
    xyt = xyt.clone().detach().requires_grad_(True)
    u = model(xyt)                    # (N,1)
    grads = torch.autograd.grad(outputs=u, inputs=xyt,
                                grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]  # (N,3)
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=xyt,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=xyt,
                               grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:, 1:2]

    if f is None:
        f_tensor = torch.zeros_like(u)
    else:
        f_tensor = f.clone().detach()
        if f_tensor.dim() == 1:
            f_tensor = f_tensor.unsqueeze(1)
        f_tensor = f_tensor.to(u.device)

    residual = u_t - alpha * (u_xx + u_yy) - f_tensor
    return torch.mean(residual ** 2), f_tensor

def boundary_loss_dirichlet(model, xyt_boundary, u_boundary=None):
    if u_boundary is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    u_pred = model(xyt_boundary)
    if u_boundary.dim() == 1:
        u_boundary = u_boundary.unsqueeze(1)
    return torch.mean((u_pred - u_boundary.to(u_pred.device))**2)

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
    return torch.mean((a*u_boundary + b*du_dn - c)**2)

# einfache Randpunkt-Generierung (Dirichlet)
def generate_boundary_points(x_min, x_max, y_min, y_max, t_min, t_max, n_per_side=50):
    t_left = torch.rand(n_per_side,1)*(t_max-t_min)+t_min
    y_left = torch.rand(n_per_side,1)*(y_max-y_min)+y_min
    x_left = torch.full_like(y_left, x_min)
    left = torch.cat([x_left, y_left, t_left], dim=1)

    t_right = torch.rand(n_per_side,1)*(t_max-t_min)+t_min
    y_right = torch.rand(n_per_side,1)*(y_max-y_min)+y_min
    x_right = torch.full_like(y_right, x_max)
    right = torch.cat([x_right, y_right, t_right], dim=1)

    t_bottom = torch.rand(n_per_side,1)*(t_max-t_min)+t_min
    x_bottom = torch.rand(n_per_side,1)*(x_max-x_min)+x_min
    y_bottom = torch.full_like(x_bottom, y_min)
    bottom = torch.cat([x_bottom, y_bottom, t_bottom], dim=1)

    t_top = torch.rand(n_per_side,1)*(t_max-t_min)+t_min
    x_top = torch.rand(n_per_side,1)*(x_max-x_min)+x_min
    y_top = torch.full_like(x_top, y_max)
    top = torch.cat([x_top, y_top, t_top], dim=1)

    Xyt_b = torch.cat([left, right, bottom, top], dim=0)
    return Xyt_b

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

# -------------------------
# Initial condition (IC) vorbereiten
# -------------------------
x_ic = xyt.get_x_ic().flatten()
y_ic = xyt.get_y_ic().flatten()
t_ic = xyt.get_t_ic().flatten()
Xyt_ic = torch.stack([x_ic, y_ic, t_ic], dim=1).float().to(device)  # (nx*ny, 3)
u_ic = initial_func(Xyt_ic).to(device)  # (N,1)

# -------------------------
# 3) Trainingsparameter für Continuity-Loss
# -------------------------
eps_time = 1e-3   # kleines Zeit-Offset, prüfe auch größere Werte (1e-2)

# Erzeuge Xyt_ic_eps einmal (t = eps_time)
Xyt_ic_eps = Xyt_ic.clone().detach()
Xyt_ic_eps[:,2] = eps_time
Xyt_ic_eps = Xyt_ic_eps.to(device)

# -------------------------
# Boundary points (Dirichlet = 0 example)
# -------------------------
# xyt_boundary = generate_boundary_points(0.0, lx, 0.0, ly, 0.0, 1.0, n_per_side=200).float().to(device)
xyt_boundary, normals = generate_boundary_points_and_normals(0.0, lx, 0.0, ly, 0.0, 1.0, n_per_side=200)
xyt_boundary = xyt_boundary.float().to(device)
u_boundary_target = 0.1*torch.ones(xyt_boundary.shape[0], 1, device=device)  # Dirichlet u=0 on boundary

# -------------------------
# Training
# -------------------------
epochs = 5000
lambda_phy = 1.0
lambda_ic = 10.0
lambda_bc = 1.0
lambda_cont = 10.0

import time

start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    # sample interior points
    # xyt_batch = sample_interior_points(XYT_all, n_samples=4096).to(device)
    xyt_batch = sample_interior_points_biased(XYT_all, n_samples=4096).to(device)

    # Spalte 2 (Index 2) ist t
    mask = XYT_all[:, 2] == 0       # Bool-Maske für t==0
    XYT_t0 = XYT_all[mask]          # Nur Zeilen mit t==0

    # physics loss: f=None (homogene PDE). Wenn du eine Quelle willst, pass f=(N,1)
    loss_phy, heat_s2 = physics_loss(model, xyt_batch, alpha, heat_source(xyt_batch))

    # initial condition loss (enforce u(x,y,t=0)=gaussian)
    u_pred_ic = model(Xyt_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    u_pred_ic_eps = model(Xyt_ic_eps)
    loss_cont = torch.mean((u_pred_ic_eps - u_ic)**2)

    # loss_bc = boundary_loss_dirichlet(model, xyt_boundary, 0*u_boundary_target)
    loss_bc = boundary_loss_robin(model, xyt_boundary, normals, 0.0, 1.0, 0*u_boundary_target)

    # loss = lambda_phy*loss_phy + lambda_ic*loss_ic + lambda_bc*loss_bc
    loss = lambda_phy*loss_phy + lambda_ic*loss_ic + lambda_bc*loss_bc + lambda_cont*loss_cont
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == epochs-1:
        # Min/Max vom Modell auf Innenpunkten:
        with torch.no_grad():
            u_pred_batch = model(xyt_batch)
            u_min = u_pred_batch.min().item()
            u_max = u_pred_batch.max().item()

            u_bc_pred = model(xyt_boundary)
            u_bc_min = u_bc_pred.min().item()
            u_bc_max = u_bc_pred.max().item()
            u_bc_mean = u_bc_pred.mean().item()
            # u_bc_rms = u_bc_pred.rms().item()

        xyt_small = Xyt_ic.clone()
        
        xyt_small[:,2]=1e-3
        xyt_small.requires_grad_(True)
        u_t_pred = torch.autograd.grad(model(xyt_small), xyt_small,
                                       grad_outputs=torch.ones_like(model(xyt_small)),
                                       create_graph=True)[0][:,2]
        print(f"Laufzeit: {time.time()-start_time:.4f} Sekunden")
        print(f"u_t_pred[:10] : ")
        print({u_t_pred[:10]})

        print(f"Epoch {epoch:5d}: total={loss.item():.6e}, "
              f"phy={loss_phy.item():.6e}, ic={loss_ic.item():.6e}, bc={loss_bc.item():.6e}, "
              f"u_min={u_min:.4f}, u_max={u_max:.4f}, u_bc_min={u_bc_min:.4f}, u_bc_max={u_bc_max:.4f}, u_bc_mean={u_bc_mean:.4f}, u_t_pred[:10] : {u_t_pred[:10]}")
        

print("Training completed!")

# -------------------------
# Prediction
# -------------------------

# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Abfragepunkte fürs Visualisieren
# -------------------------
n_vis = 100  # Auflösung
x_vis = torch.linspace(0, lx, n_vis)
y_vis = torch.linspace(0, ly, n_vis)
t_vis = torch.linspace(0, 1.0, 100)
Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')

# shape für 2D-Darstellung
nx_ic = xyt.get_x_ic().shape[0]
ny_ic = xyt.get_y_ic().shape[1]

# u_ic_frame1 = u_ic.detach().cpu().reshape(nx_ic, ny_ic).numpy()
u_ic_frame = initial_func(torch.stack([Xv.flatten(), Yv.flatten()], dim=1).float()).detach().cpu().reshape(n_vis, n_vis).numpy()

u_frames = [u_ic_frame,  ] 
u_means = []
with torch.no_grad():
    for tval in t_vis:
        Xyt_vis = torch.stack([
            Xv.flatten(),
            Yv.flatten(),
            torch.full_like(Xv.flatten(), tval)
        ], dim=1).to(device)
        u_pred = model(Xyt_vis).reshape(n_vis, n_vis).cpu().numpy()
        u_frames.append(u_pred)
        u_mean = u_pred.mean()
        u_means.append(u_mean)
        print(f"Frame {tval*100:.0f}: u mean={u_mean:.6f}, ")

print(f"min of u mean={np.min(np.array(u_means))}")
print(f"max of u mean={np.max(np.array(u_means))}")
print(f"Standard dev of u mean={np.std(np.array(u_means))}")

u_frames = np.array(u_frames)  # shape (nt, nx, ny)

for idx in [0,1,-1]:
    results = u_frames
    nt_vis, nx, ny = results.shape
    fig, ax = plt.subplots(figsize=(8,6))
    plt.subplots_adjust(bottom=0.25)

    vmin, vmax = results.min(), results.max()
    cax = ax.imshow(results[idx], origin='lower', extent=[0, lx, 0, ly],
                    cmap='coolwarm', vmin=vmin, vmax=vmax)
    fig.colorbar(cax, label="Temperature")
    ax.set_title(f"Frame: {idx}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# -------------------------
# Einzelbild (z.B. erstes t)
# -------------------------
plt.figure(figsize=(6,5))
plt.imshow(u_frames[0], origin='lower', extent=[0, lx, 0, ly], cmap='coolwarm')
plt.colorbar(label="Temperature")
plt.title("Temperature at start")
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