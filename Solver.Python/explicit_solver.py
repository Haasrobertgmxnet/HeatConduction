import numpy as np
import time

from boundary_conditions import HeatBoundaryCondition
from result_data import result_data
from result_frames import result_frames

def D2_x(u, dx=1.0):
    """
    Second derivative in x direction
    Parameters
    ----------
    u : ndarray of shape (nx, ny)
        2D scalar field.
    dx, dy : float
        Grid spacing in x direction.

    Returns
    -------
    ndarray of shape (nx-2, ny-2)
        2D field of all second derivatives in in x direction.
    """
    return (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dx

def D2_y(u, dy = 1.0):
    """
    Second derivative in y direction
    Parameters
    ----------
    u : ndarray of shape (nx, ny)
        2D scalar field.
    dx, dy : float
        Grid spacing y direction.

    Returns
    -------
    ndarray of shape (nx-2, ny-2)
        2D field of all second derivatives in in y direction.
    """
    return (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dy

class HeatExplicitSolver():
    def __init__(self, alpha, dx, dy, dt, bc):
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
        self.current_D2_x = None
        self.current_D2_y = None
        self.current_u = None
        self.current_pde_loss = None

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
        u_new = u.copy()
        self.current_D2_x = D2_x(u)
        self.current_D2_y = D2_y(u)
        u_new[1:-1, 1:-1] = (u[1:-1, 1:-1]
        + self.lamx * self.current_D2_x
        + self.lamy * self.current_D2_y + self.dt*f[1:-1, 1:-1])
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
        pde_res = None
        u_t = None
        for _ in range(nt):
            u_old = u
            u = self.step(u,f)
            u = self.apply_bc(u, self.dx, self.dy)
            u = self.step(u, f)
            u_t = (u - u_old) / self.dt
            pde_res =  (self.lamx * self.current_D2_x + self.lamy * self.current_D2_y + self.dt*f[1:-1, 1:-1] - (u[1:-1, 1:-1] - u_old[1:-1, 1:-1]))/self.dt

        self.current_u = u
        self.current_pde_loss = pde_res
        return u, u_t

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
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
        f = ibvp.heat_source(xy[:,0], xy[:,1])
        f = f.reshape(ny, nx)

        neumann_bc = HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c)
        dx, dy = lx/(nx-1), ly/(ny-1)
        solver = HeatExplicitSolver(ibvp.alpha, dx, dy, dt, neumann_bc.apply)
        if not solver.check_stability():
            print("CFL condition violated")

        u_frames = [result_data(u0)]
        u = u0.copy()
        for n_frame in range(n_frames):
            start = time.time()
            tval = frame.lt*(1+n_frame)/n_frames
            u, u_t = solver.n_steps(u, f, t_steps_per_frame)
            u_frames.append(result_data(u, u_t))
            min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
            max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
            tval = (n_frame + 1) * (frame.lt / n_frames)
            print(f"Frame {tval:.2f}: mean={u.mean():.6f}, min={u.min():.6f} @ {min_idx}, max={u.max():.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")

        result = result_frames(u_frames, f, has_u_t= False, has_derivs= False, has_laplacian= False)
        return result
