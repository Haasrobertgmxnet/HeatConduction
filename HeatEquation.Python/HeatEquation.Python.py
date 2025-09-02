import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve

# Parameters
nx, ny = 30, 30        # grid points
lx, ly = 1.0, 1.0      # domain size
dx, dy = lx/nx, ly/ny
alpha = 0.01           # diffusion coefficient
dt = 0.0005            # time step
nt = 200               # number of time steps
nts = [0, 50, 100, 150, 200] # number of time steps

# Initial condition: hot spot in the middle
u0 = np.zeros((nx, ny))
u0[nx//2, ny//2] = 1.0

# Explicit Euler scheme for one step
def heat_explicit_step(u, alpha, dx, dy, dt):
    u = u.copy()
    lamx = alpha * dt / dx**2
    lamy = alpha * dt / dy**2
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]
    + lamx * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])
    + lamy * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]))
    return u

# Explicit Euler scheme for nt steps
def heat_explicit(u, alpha, dx, dy, dt, nt):
    # un = u.copy()
    for _ in range(nt):
        u = heat_explicit_step(u, alpha, dx, dy, dt)
    return u

def Crank_Nicolson_matrices(u, alpha, dx, dy, dt):
    u = u.copy().reshape(-1)
    lamx = alpha * dt / (2*dx**2)
    lamy = alpha * dt / (2*dy**2)

    # 1D second difference matrices
    ex = np.ones(nx)
    ey = np.ones(ny)
    Tx = diags([ex, -2*ex, ex], [-1, 0, 1], shape=(nx, nx))
    Ty = diags([ey, -2*ey, ey], [-1, 0, 1], shape=(ny, ny))

    # 2D Laplacian using Kronecker products
    Ix = eye(nx)
    Iy = eye(ny)
    L = kron(Iy, Tx)/(dx**2) + kron(Ty, Ix)/(dy**2)

    # Crank Nicolson matrices
    A = eye(nx*ny) - alpha*dt/2 * L
    B = eye(nx*ny) + alpha*dt/2 * L

    return [A, B]

# Function: implicit Crank Nicolson scheme for one step
def heat_implicit_step(u, A = None, B = None):
    if A is None or B is None:
        global alpha, dx, dy, dt
        [A,B] = Crank_Nicolson_matrices(u, alpha, dx, dy, dt)
    u = u.copy().reshape(-1)
    u = spsolve(A, B @ u)
    return u.reshape((nx, ny))

# Function: implicit Crank Nicolson scheme fot nt steps
def heat_implicit(u, alpha, dx, dy, dt, nt):
    u = u.copy().reshape(-1)
    [A,B] = Crank_Nicolson_matrices(u, alpha, dx, dy, dt)

    for n in range(nt):
        u = heat_implicit_step(u, A, B)

    return u.reshape((nx, ny))




def example1():
    # Run explicit and implicit schemes
    ue = heat_explicit(u0, alpha, dx, dy, dt, nt)
    ui = heat_implicit(u0, alpha, dx, dy, dt, nt)

    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(ue, origin="lower", cmap="hot")
    axs[0].set_title("Explicit Euler")
    axs[1].imshow(ui, origin="lower", cmap="hot")
    axs[1].set_title("Implicit Crank Nicolson")
    plt.show()

def example2(alpha):
    # Prepare animation
    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(u0, origin="lower", cmap="coolwarm", animated=True) # or cmap="viridis"
    axs[0].set_title("Heat diffusion")

    im1 = axs[1].imshow(u0, origin="lower", cmap="coolwarm", animated=True) # or cmap="viridis"
    axs[1].set_title("Heat diffusion")


    u = u0.copy()


    def update(frame):
        nonlocal u  # refers to u from example2 scope
        nonlocal alpha
        u = heat_explicit_step(u, alpha, dx, dy, dt)
        im0.set_array(u)
        axs[0].set_title(f"Explicit Euler. timestep {frame}")

        [A,B] = Crank_Nicolson_matrices(u, alpha, dx, dy, dt)
        u = heat_implicit_step(u, A, B)
        im1.set_array(u)
        axs[1].set_title(f"Crank Nicolson. timestep {frame}")
        return [im0, im1]


    ani = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)
    plt.show()

def example2_compare_alphas(alphas):
    """
    Compare Explicit Euler vs Crank-Nicolson animations for multiple alpha values
    """
    for alpha_val in alphas:
        print(f"Running animation for alpha = {alpha_val}")
        # Prepare animation
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axs[0].imshow(u0, origin="lower", cmap="coolwarm", animated=True)
        axs[0].set_title("Explicit Euler")

        im1 = axs[1].imshow(u0, origin="lower", cmap="coolwarm", animated=True)
        axs[1].set_title("Crank Nicolson")

        u_explicit = u0.copy()
        u_cn = u0.copy()

        [A,B] = Crank_Nicolson_matrices(u_cn, alpha_val, dx, dy, dt)

        def update(frame):
            nonlocal u_explicit, u_cn
            # Explicit step
            u_explicit = heat_explicit_step(u_explicit, alpha_val, dx, dy, dt)
            im0.set_array(u_explicit)
            axs[0].set_title(f"Explicit Euler, alpha {alpha_val}, timestep {frame}")

            # Crank-Nicolson step
            u_cn = heat_implicit_step(u_cn, A, B)
            im1.set_array(u_cn)
            axs[1].set_title(f"Crank Nicolson, alpha {alpha_val}, timestep {frame}")

            return [im0, im1]

        ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
        plt.show()


def main() -> None:
    """
    Main execution function that compares explicit and implicit solvers for the 2D heat equation
    """
    # example2(alpha)
    example2_compare_alphas([0.001, 0.01, 0.05, 0.1])

if __name__ == "__main__":
    main()
