import numpy as np
from numba import njit
from boundary_conditions import HeatBoundaryCondition
import time

@njit
def step_numba(u, lamx, lamy, dt, f):
    """
    Perform one explicit Euler update step for the 2D heat equation using finite differences.

    Parameters
    ----------
    u : ndarray of shape (nx, ny)
        Current temperature field.
    lamx, lamy : float
        Diffusion CFL factors alpha*dt/dx^2 and alpha*dt/dy^2.
    dt : float
        Time step size.
    f : ndarray of shape (nx, ny)
        Source term evaluated on grid.

    Returns
    -------
    u_new : ndarray of shape (nx, ny)
        Updated temperature field at next time step.

    Notes
    -----
    Boundary values are not updated here. They must be applied externally.
    """
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
        """
        Explicit finite difference heat solver (Euler forward in time).

        Parameters
        ----------
        alpha : float
            Diffusion coefficient in u_t = alpha (u_xx + u_yy) + f.
        dx, dy : float
            Grid spacings in x and y directions.
        dt : float
            Time step size.
        bc : callable
            Boundary condition apply function: bc(u, dx, dy) -> u.
        use_numba : bool, optional
            Whether to use JIT-compiled stencil update for speed.
        """
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
        """
        Check the explicit scheme stability condition:
            lamx + lamy <= 0.5

        Returns
        -------
        bool
            True if stable, False if CFL condition is violated.
        """
        stability_number = self.lamx + self.lamy
        return stability_number <= 0.5

    def step(self, u, f = None):
        """
        Compute one explicit Euler time step (no boundary conditions applied here).

        Parameters
        ----------
        u : ndarray of shape (nx, ny)
            Current temperature field.
        f : ndarray of shape (nx, ny), optional
            Source field. If None, assumed zero.

        Returns
        -------
        u_new : ndarray of shape (nx, ny)
            Updated state after one time step.
        """
        if f is None:
            f = 0*u
        if self.use_numba:
            return step_numba(u, self.lamx, self.lamy, self.dt, f)
        u_new = u.copy()
        u_new[1:-1, 1:-1] = (u[1:-1, 1:-1]
        + self.lamx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])
        + self.lamy * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) + self.dt*f[1:-1, 1:-1])
        return u_new

    def n_steps(self, u, f = None, nt= 1):
        """
        Perform nt explicit time steps and apply boundary conditions after each step.

        Parameters
        ----------
        u : ndarray (nx, ny)
            Initial field.
        f : ndarray (nx, ny), optional
            Source term.
        nt : int
            Number of time steps.

        Returns
        -------
        u : ndarray (nx, ny)
            Field after nt steps.
        """
        for _ in range(nt):
            u = self.step(u,f)
            u = self.apply_bc(u, self.dx, self.dy)
        return u

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1, use_numba= False):
        """
        High-level time simulation pipeline for producing successive solution frames.

        Parameters
        ----------
        ibvp : object
            Problem specification providing:
              - alpha
              - a, b, c boundary coefficients
              - initial_u(x,y)
              - heat_source(x,y)
        frame : object
            Frame/grid configuration providing:
              - nx, ny : grid size
              - lx, ly : domain size
              - nt : total time steps
              - lt : final simulation time
        t_steps_per_frame : int
            Number of update steps per output frame.
        n_frames : int
            Number of simulation frames to compute.
        use_numba : bool
            Whether to accelerate using numba JIT.

        Returns
        -------
        frames : list of ndarray
            Sequence of solution fields at output times.
        u_means : list of float
            Mean temperature per frame.
        """
        print("Explicit solver")
        nx, ny = frame.nx, frame.ny
        lx, ly = frame.lx, frame.ly
        nt = frame.nt
        dt = frame.lt/nt

        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y, indexing='xy')
        xy = np.column_stack([X.ravel(), Y.ravel()])
        u0 =ibvp.initial_u(xy[:,0], xy[:,1])
        u0 = u0.reshape(ny, nx)
        h = ibvp.heat_source(xy[:,0], xy[:,1])
        h = h.reshape(ny, nx)

        neumann_bc = HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c)
        dx, dy = lx/(nx-1), ly/(ny-1)
        solver = HeatExplicitSolver(ibvp.alpha, dx, dy, dt, neumann_bc.apply, use_numba)
        if not solver.check_stability():
            print("CFL condition violated")

        frames = [u0,]
        u_means = []
        u = u0.copy()
        for n_frame in range(n_frames):
            start = time.time()
            tval = frame.lt*(1+n_frame)/n_frames
            u = solver.n_steps(u, h, t_steps_per_frame)
            frames.append(u)
            u_mean = u.mean()
            u_min = u.min()
            u_max = u.max()
            min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
            max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
            u_means.append(u_mean)
            tval = (n_frame + 1) * (frame.lt / n_frames)
            print(f"Frame {tval:.2f}: mean={u_mean:.6f}, min={u_min:.6f} @ {min_idx}, max={u_max:.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")

        return frames, u_means
