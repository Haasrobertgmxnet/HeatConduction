import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve

# -------------------------------------------------
# Grid
nx, ny = 30, 30
lx, ly = 1.0, 1.0
dx, dy = lx/nx, ly/ny
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

# Physikalische Parameter
alpha = 0.01
dt = 0.0005
nt = 200

# -------------------------------------------------
# Anfangsbedingung
u = np.zeros((nx, ny))
sigma = 0.1
for i in range(nx):
    for j in range(ny):
        u[i,j] = 50.0*np.exp(-((x[i]-lx/2)**2+(y[j]-ly/2)**2)/(2*sigma**2))

# -------------------------------------------------
# Laplace-Matrix in x und y Richtung
rx = alpha*dt/(2*dx*dx)
ry = alpha*dt/(2*dy*dy)

# 1D-Laplace-Matrix für x
ex = np.ones(nx)
Ax = diags([ex[:-1], -2*ex, ex[:-1]], [-1,0,1], shape=(nx,nx))
# 1D-Laplace-Matrix für y
ey = np.ones(ny)
Ay = diags([ey[:-1], -2*ey, ey[:-1]], [-1,0,1], shape=(ny,ny))

# 2D-Laplace über Kroneckerprodukte
Lx = kron(identity(ny), Ax)
Ly = kron(Ay, identity(nx))
L = rx*Lx + ry*Ly  # zusammengesetzter Operator

I = identity(nx*ny)
A_mat = (I - L)   # linke Matrix
B_mat = (I + L)   # rechte Matrix

# -------------------------------------------------
# Dirichlet-Randbedingungen: u=0 außen
def apply_bc_vec(u_vec):
    U = u_vec.reshape(nx,ny)
    U[0,:] = 0
    U[-1,:] = 0
    U[:,0] = 0
    U[:,-1] = 0
    return U.ravel()

# -------------------------------------------------
# Zeitintegration Crank-Nicolson
u_vec = u.ravel()
results = [u.copy()]

for n in range(nt):
    rhs = B_mat.dot(u_vec)
    # Randbedingungen einbauen:
    Utemp = apply_bc_vec(rhs)
    rhs = Utemp
    u_vec = spsolve(A_mat, rhs)
    U = u_vec.reshape(nx,ny)
    U[0,:] = 0
    U[-1,:] = 0
    U[:,0] = 0
    U[:,-1] = 0
    results.append(U.copy())
    u_vec = U.ravel()

results = np.array(results)  # shape (nt+1, nx, ny)

# -------------------------------------------------
# Visualisierung
plt.figure(figsize=(6,5))
plt.imshow(results[-1], origin='lower', extent=[0,lx,0,ly], cmap='hot')
plt.colorbar(label='Temperature')
plt.title('Crank-Nicolson Temperatur am Endzeitpunkt')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Peak über die Zeit plotten
peak = [r[nx//2,ny//2] for r in results]
plt.figure()
plt.plot(np.arange(nt+1)*dt, peak, '-o')
plt.xlabel('time')
plt.ylabel('Peak Temperature')
plt.title('Peak Decay - Crank-Nicolson')
plt.grid()
plt.show()

# -------------------------
# Animation or slider
# -------------------------
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

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

