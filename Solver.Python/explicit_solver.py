import numpy as np
from numba import njit

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

    def check_stability(self):
        stability_number = self.lamx + self.lamy
        return stability_number <= 0.5, stability_number

    def step(self, u, f = None):
        if f is None:
            f = 0*u
        if self.use_numba:
            return step_numba(u, self.lamx, self.lamy, self.dt, f)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (u[1:-1, 1:-1]
        + self.lamx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        + self.lamy * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) - self.dt*f[1:-1, 1:-1])
        return u_new

    # Explicit Euler scheme for nt steps
    def n_steps(self, u, f = None, nt= 1):
        for _ in range(nt):
            u = self.step(u,f)
            self.apply_bc(u, self.dx, self.dy)
        return u


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


