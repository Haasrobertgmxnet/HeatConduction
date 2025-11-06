import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import time

class GreenFunctionSolver:
    def __init__(self, alpha, bc, Lx=1.0, Ly=1.0, M=20, N=20):
        """
        Initialize the Green function solver for a 2D heat equation with Robin-type boundary conditions.

        Parameters
        ----------
        alpha : float
            Diffusion coefficient in the PDE u_t = alpha * (u_xx + u_yy) + f.
        bc : object
            Boundary condition object providing attributes a, b, c.
            Represents boundary constraints of the form a*u + b*(du/dn) = c.
        Lx, Ly : float, optional
            Domain size in x and y directions. (Currently logic assumes unit domain.)
        M, N : int, optional
            Number of eigenmodes in x and y (truncation size). (Currently fixed to stored arrays.)
        """
        print(f"Gesuchter Wert: {0.48503638*204/214}")
        self.alpha = alpha
        self.apply_bc = bc

        # Precomputed eigenvalues for separated Laplacian eigenmodes
        self.eig_vals = np.array([ 0.96018887, 3.43101431, 6.43819715, 9.52961783, 12.64540952,
         15.77134816, 18.90244679, 22.03652001, 25.17246196, 28.30965385, 31.44772266,
         34.58643025, 37.72561748, 40.86517399, 44.00502085, 47.14510012, 50.2853683,
         53.42579212, 56.56634566])

        # Scaling coefficients corresponding to eigenmodes
        self.scal = np.array([0.48503638, 0.1087893,  0.04307502, 0.0239897, 0.01570866,
         0.01128258, 0.00860051, 0.00683343, 0.00559754, 0.0046936, 0.00400903,
         0.00347597, 0.00305132, 0.00270657, 0.00242215, 0.00218426, 0.00198289,
         0.00181066, 0.00166199])

    def phi(self, eig_vals, x):
        """
        Compute separated spatial eigenfunctions phi_k(x) satisfying the boundary conditions.

        Parameters
        ----------
        eig_vals : array-like of shape (M,)
            Eigenvalues k associated with separated Laplace operator modes.
        x : float or array-like of shape (Nx,) or broadcastable
            Spatial coordinate(s) where phi is evaluated.

        Returns
        -------
        phi_vals : ndarray of shape (M, Nx)
            Matrix where each row corresponds to phi_k(x) for one eigenmode.

        Notes
        -----
        Defines mode shapes:
            phi_k(x) = sin(k x) + (k/gamma) * cos(k x),
        where gamma = a / b from the boundary condition.
        """
        mode = np.atleast_1d(eig_vals)[:, None]
        x = np.atleast_1d(x)[None, :]
        gamma = self.apply_bc.a / self.apply_bc.b
        phi_vals = np.sin(mode * x) + (mode/gamma)*np.cos(mode * x)

        debug_ = False
        if not debug_:
            return phi_vals
        # Diagnostic output if needed
        arg = mode * x
        print(f"Size x {x.size}")
        print(f"Size mode {mode.size}")
        print(f"Size arg {arg.size}")
        print(f"Size phi: {phi_vals.size}")
        return phi_vals

    def green(self, x, y, x0, y0, tau):
        """
        Compute the Green function G(x,y,tau; x0,y0) and integrated kernel G_int
        for the 2D heat equation using truncated separation of variables.

        Parameters
        ----------
        x, y : float or array-like
            Coordinates where the solution is evaluated. May be scalars or 1D grids.
        x0, y0 : float or array-like (broadcastable to match x,y evaluation loops)
            Source coordinates in the domain.
        tau : float
            Time difference t - s (always non-negative). Represents heat propagation time.

        Returns
        -------
        G : ndarray of shape (len(x), len(y))
            Green function kernel applied to initial condition.
        G_int : ndarray of shape (len(x), len(y))
            Time-integrated Green function for convolution with source term f.

        Notes
        -----
        Uses a rank-M spectral approximation:
            G(x,y,t; x0,y0) = sum_{m,n} phi_m(x) phi_m(x0) phi_n(y) phi_n(y0)
                              * exp(-alpha (k_m^2 + k_n^2) t)
        """
        if tau < 0:
            return np.zeros((len(x), len(y)))

        eig_vals = self.eig_vals
        scal = self.scal

        k = np.atleast_1d(eig_vals)

        xs = ys = np.linspace(0.,1.,20)
        PHI_x = self.phi(k, xs)
        PHI_y = self.phi(k, ys)
        dx = np.mean(np.diff(xs))
        dy = np.mean(np.diff(ys))

        variant1 = False
        if variant1:
            norms_x = np.sqrt(np.sum(PHI_x**2, axis=1)*dx)
            norms_y = np.sqrt(np.sum(PHI_y**2, axis=1)*dy)
            scal = 1/norms_x

        PHI_x_norm = self.phi(k, x)*scal[:,None]
        PHI_x0_norm = self.phi(k, x0)*scal[:,None]
        PHI_y_norm = self.phi(k, y)*scal[:,None]
        PHI_y0_norm = self.phi(k, y0)*scal[:,None]

        km2 = k**2
        ex = self.alpha * (km2[:,None] + km2[None,:])
        A = np.exp(-ex * tau)
        A_int = 1/ex*(1-A)

        C = A @ (PHI_y0_norm[:,0][:,None] * PHI_y_norm)
        D = (PHI_x0_norm[:,0][:,None] * PHI_x_norm)
        G = D.T @ C

        C_int = A_int @ (PHI_y0_norm[:,0][:,None] * PHI_y_norm)
        D = (PHI_x0_norm[:,0][:,None] * PHI_x_norm)
        G_int = D.T @ C_int

        return G, G_int

    def u(self, x, y, t, u0_func, f_func=None):
        """
        Compute solution u(x,y,t) using Green function convolution.

        Parameters
        ----------
        x, y : array-like
            Target grid coordinates for evaluating the solution.
        t : float
            Time at which solution is evaluated.
        u0_func : callable u0(x,y)
            Initial condition function.
        f_func : callable f(x,y), optional
            Source term function in the PDE. If None, homogeneous equation assumed.

        Returns
        -------
        U : ndarray of shape (len(x), len(y))
            Solution field at time t.
        """
        # --- Prepare cache for projections and Base (once) ---
        if not hasattr(self, "_proj_cache"):
            xs = np.linspace(0, 1, 101)
            ys = np.linspace(0, 1, 101)
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]

            k = self.eig_vals
            phi_x = self.phi(k, xs) * self.scal[:, None]   # (M, Nx)
            phi_y = self.phi(k, ys) * self.scal[:, None]   # (M, Ny)

            self._proj_cache = {
                "xs": xs, "ys": ys, "dx": dx, "dy": dy,
                "phi_x": phi_x, "phi_y": phi_y,
            }

        xs = self._proj_cache["xs"]; ys = self._proj_cache["ys"]
        dx = self._proj_cache["dx"]; dy = self._proj_cache["dy"]
        phi_x = self._proj_cache["phi_x"]; phi_y = self._proj_cache["phi_y"]

        U_ofs = self.apply_bc.c / self.apply_bc.a
        k = self.eig_vals
        lam = k**2
        L = (lam[:, None] + lam[None, :])                  # (M, M)

        # --- projection of the initial state (caching once) ---
        if not hasattr(self, "_C0"):
            X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
            F0 = (u0_func(X0, Y0) - U_ofs)                  # (Nx, Ny)
            self._C0 = phi_x @ F0 @ phi_y.T * dx * dy       # (M, M)

        # --- projection to heat source (f time-independent) (caching once) ---
        if f_func is not None and not hasattr(self, "_Cf_static"):
            X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
            try:
                F = f_func(X0, Y0)                          # preferred: f(x,y)
            except TypeError:
                F = f_func(X0, Y0, 0.0)                     # if signature f(x,y,t)
            self._Cf_static = phi_x @ F @ phi_y.T * dx * dy # (M, M)

        # --- time factors ---
        decay = np.exp(-self.alpha * L * t)                 # (M, M)

        C = self._C0 * decay                                # Anfangsanteil

        if f_func is not None:
            # closed form for constant f(x,y):
            # integral_0^t e^{-α L (t-s)} ds = (1 - e^{-α L t}) / (α L)
            C += self._Cf_static * (1.0 - decay) / (self.alpha * L + 1e-300)

        # --- reconstruction at target grid ---
        phi_x_eval = self.phi(k, x) * self.scal[:, None]    # (M, Nx_eval)
        phi_y_eval = self.phi(k, y) * self.scal[:, None]    # (M, Ny_eval)
        U = phi_x_eval.T @ C @ phi_y_eval                   # (Nx_eval, Ny_eval)

        return U_ofs + U

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
        """
        Helper routine that evolves solution in time and collects frames.
        
        Parameters
        ----------
        ibvp : object
            Problem specification providing:
              - initial_u(x, y)
              - heat_source(x, y, t) (not used here in prediction mode)
              - alpha, a, b, c etc. (not directly used in inference)
        frame : object
            Grid and time settings:
              - nx, ny : number of spatial grid points
              - lx, ly : physical dimensions of the domain
              - nt : number of time steps
              - lt : final time
        t_steps_per_frame : int
            Unused here (included only for API consistency).
        n_frames : int
            Number of time values for which to evaluate and return solution frames.

        Returns
        -------
        u_frames : list of ndarray
            List of predicted temperature fields, each shaped (ny, nx).
        u_means : list of float
            Mean temperature value per predicted frame.
        """

        print("Green function")
        nx, ny = frame.nx, frame.ny
        lx, ly = frame.lx, frame.ly
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)

        solver = GreenFunctionSolver(ibvp.alpha, ibvp, 1.0, 1.0)

        frames = [ibvp.initial_u,]
        u_means = []
        
        for n_frame in range(n_frames):
            start = time.time()
            tval = frame.lt*(1+n_frame)/n_frames
            u = solver.u(x,y,tval,ibvp.initial_u,ibvp.heat_source)
            frames.append(u)
            u_mean = u.mean()
            u_min = u.min()
            u_max = u.max()
            min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
            max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
            u_means.append(u_mean)
            print(f"Frame {tval:.2f}: mean={u_mean:.6f}, min={u_min:.6f} @ {min_idx}, max={u_max:.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")

        return frames, u_means
