import numpy as np
from numba import njit
from boundary_conditions import HeatBoundaryCondition
import torch
import time

@njit
def step_numba(u, lamx, lamy, dt, f):
    nx, ny = u.shape
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = (
                u[i, j]
                + lamx * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
                + lamy * (u[i, j+1] - 2*u[i, j] + u[i, j-1])
                - dt * f[i, j]
            )
    return u_new

class HeatExplicitSolver():
    def __init__(self, alpha, dx, dy, dt, bc, use_numba = False):
        self.alpha= alpha
        self.dt= dt
        self.lamx = self.alpha * self.dt / dx**2
        self.lamy = self.alpha * self.dt / dy**2
        self.dx = dx
        self.dy = dy
        self.apply_bc = bc
        self.use_numba = use_numba
        print(f"alp: {alpha:.5}")

    def check_stability(self):
        stability_number = self.lamx + self.lamy
        return stability_number <= 0.5

    def step(self, u, f = None):
        if f is None:
            f = 0*u
        if self.use_numba:
            return step_numba(u, self.lamx, self.lamy, self.dt, f)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (u[1:-1, 1:-1]
        + self.lamx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        + self.lamy * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) + self.dt*f[1:-1, 1:-1])
        return u_new

    # Explicit Euler scheme for nt steps
    def n_steps(self, u, f = None, nt= 1):
        for _ in range(nt):
            u = self.step(u,f)
            u = self.apply_bc(u, self.dx, self.dy)
        return u

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1, use_numba= False):
        
        nx, ny = frame.nx, frame.ny
        lx, ly = frame.lx, frame.ly
        nt = frame.nt
        dt = frame.lt/nt

        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y, indexing='xy')  # X,Y sind shape (ny, nx)
        xy = np.column_stack([X.ravel(), Y.ravel()])
        u0 =ibvp.initial_u(xy[:,0], xy[:,1])
        u0 = u0.reshape(ny, nx)
        h = ibvp.heat_source(xy[:,0], xy[:,1])
        h = h.reshape(ny, nx)

        neumann_bc = HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c)
        dx, dy = lx/nx, ly/ny
        solver = HeatExplicitSolver(ibvp.alpha, dx, dy, dt, neumann_bc.apply_robin, use_numba)
        if not solver.check_stability():
            print("CFL condition violated")

        frames = [u0,]
        u_means = []
        u = u0.copy()
        for n_frame in range(n_frames):
            tval = frame.lt*(1+n_frame)/n_frames
            u = solver.n_steps(u, h, t_steps_per_frame)
            frames.append(u)
            u_mean = u.mean()
            u_means.append(u_mean)
            print(f"Frame {tval:.2f}: u mean={u_mean:.6f}, ")
        return frames, u_means

############

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


