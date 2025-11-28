import os
from re import U
from tkinter import SE
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import time

from result_data import result_data
from result_frames import result_frames

from int_methods import IntegrationMethod
from int_methods import simpson_weights, three_eights_weights, milne_weights
from int_methods import clenshaw_curtis, cc_transform
from calculate_modes import func, solve, fplot

def inner_L2(f, g, a=0.0, b=1.0, n=10000):
    x = np.linspace(a, b, n, endpoint=True)
    fx = f(x)
    gx = g(x)
    # komplexer Fall: Konjugation nicht vergessen
    return np.trapz(fx * np.conjugate(gx), x)

def norm_L2(f, a=0.0, b=1.0, n=10000):
    return np.sqrt(np.real(inner_L2(f, f, a, b, n)))

def I_closed(mode, gamma):
    m = np.asarray(mode, dtype=float)
    s2_over_4m = np.where(m == 0.0, 0.5, np.sin(2*m)/(4*m))  # lim_{m->0} = 1/2
    return (0.5 - s2_over_4m
            + (1.0/(2*gamma))*(1 - np.cos(2*m))
            + (m**2/gamma**2)*(0.5 + s2_over_4m))

class GreenFunctionSolver:
    def __init__(self, alpha, ibvp, Lx=1.0, Ly=1.0, M=20, N=20):
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
        self.alpha = alpha
        self.ibvp = ibvp

        # Eigenvalues for separated Laplacian eigenmodes
        self.eig_vals = (solve(func, cutoff=4001, coarse_points=10_000, refine=True, refine_points=1_000, x_min=0, x_max=9550)[1:])
        print(f"Number of eigenvalues: {self.eig_vals.shape}")

        # Scaling coefficients corresponding to eigenmodes
        self.scal = np.array([np.sqrt(1.0/I_closed(mode, 0.5)) for mode in self.eig_vals])

        # resolution for projection grid
        self.resolution = 7801

        # use filter
        self.use_filter = True

        # quadrature method
        self.integration_method = IntegrationMethod.SIMPSON

        # Fill cache
        self.prepare_cache()

    def phi(self, x):
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
        mode = np.atleast_1d(self.eig_vals)[:, None]
        scals = np.atleast_1d(self.scal)[:, None]
        x = np.atleast_1d(x)[None, :]
        gamma = self.ibvp.a / self.ibvp.b
        phi_vals = np.sin(mode * x) + (mode/gamma)*np.cos(mode * x)

        debug_ = False
        if not debug_:
            return scals*phi_vals

        # Diagnostic output if needed
        arg = mode * x
        print(f"Size x {x.size}")
        print(f"Size mode {mode.size}")
        print(f"Size arg {arg.size}")
        print(f"Size phi: {phi_vals.size}")
        return scals*phi_vals

    def dphi(self, x):
        """
        Compute the 1st derivative of separated spatial eigenfunctions phi_k(x) satisfying the boundary conditions.

        Parameters
        ----------
        eig_vals : array-like of shape (M,)
            Eigenvalues k associated with separated Laplace operator modes.
        x : float or array-like of shape (Nx,) or broadcastable
            Spatial coordinate(s) where phi is evaluated.

        Returns
        -------
        dphi_vals : ndarray of shape (M, Nx)
            Matrix where each row corresponds to dphi_k(x) for one eigenmode.

        Notes
        -----
        Defines mode shapes:
            phi_k(x) = sin(k x) + (k/gamma) * cos(k x),
        where gamma = a / b from the boundary condition.
        """
        mode = np.atleast_1d(self.eig_vals)[:, None]
        scals = np.atleast_1d(self.scal)[:, None]
        x = np.atleast_1d(x)[None, :]
        gamma = self.ibvp.a / self.ibvp.b
        dphi_vals = mode*np.cos(mode * x) - (mode**2/gamma)*np.sin(mode * x)

        return scals*dphi_vals

    def ddphi(self, x):
        """
        Second derivative of the 1D eigenfunctions phi_k(x).
        """
        mode = np.atleast_1d(self.eig_vals)[:, None]
        scals = np.atleast_1d(self.scal)[:, None]
        x = np.atleast_1d(x)[None, :]
        gamma = self.ibvp.a / self.ibvp.b

        # phi_k(x) = sin(kx) + (k/gamma)*cos(kx)
        # phi'_k(x) = k cos(kx) - (k^2/gamma) sin(kx)
        # phi''_k(x) = -k^2 sin(kx) - (k^3/gamma) cos(kx)
        ddphi_vals = -mode**2 * np.sin(mode * x) - (mode**3/gamma)*np.cos(mode * x)

        return scals * ddphi_vals

    def prepare_cache(self):
        """
        Prepare and cache projection grid and basis evaluations for performance.
        """
        # This method is now integrated into calculate_results for caching.
         # --- 1) Prepare and cache projection grid and basis evaluations -----------
        # We create a fixed "reference" grid for projections (Galerkin coefficients).
        # This is done once and cached for performance.
        if not hasattr(self, "_proj_cache"):

            xs = np.linspace(0, 1., self.resolution)   # reference grid in x (size Nx)
            ys = np.linspace(0, 1., self.resolution)   # reference grid in y (size Ny)
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]

            cc_weight_fun = lambda n: clenshaw_curtis(n, transform=cc_transform)

            def get_weight_function(method: IntegrationMethod):
                nonlocal dx, dy
                print(f"Selecting weight function for method: {method}")
                match method:
                    case IntegrationMethod.SIMPSON:
                        print("Using Simpson weights.")
                        return simpson_weights

                    case IntegrationMethod.THREE_EIGHTS:
                        print("Using 3/8 weights.")
                        return three_eights_weights

                    case IntegrationMethod.MILNE:
                        print("Using Milne weights.")
                        return milne_weights

                    case IntegrationMethod.CLENSHAW_CURTIS:
                        print("Using Clenshaw-Curtis weights.")
                        return cc_weight_fun

                    case _:
                        raise ValueError(f"Unknown integration method: {method}")

            weight_fun = get_weight_function(self.integration_method)

            wx, x = weight_fun(len(xs))
            wy, y = weight_fun(len(ys))

            wx = wx * dx if self.integration_method != IntegrationMethod.CLENSHAW_CURTIS else wx
            wy = wy * dy if self.integration_method != IntegrationMethod.CLENSHAW_CURTIS else wy

            xs = xs if x is None else x
            ys = ys if y is None else y
            
            phi_x = self.phi(xs) # shape (M, Nx)
            phi_y = self.phi(ys) # shape (M, Ny)

            dphi_x = self.dphi(xs) # shape (M, Nx)
            dphi_y = self.dphi(ys) # shape (M, Ny)

            self._proj_cache = {
                "xs": xs, "ys": ys, "dx": dx, "dy": dy,
                "phi_x": phi_x, "phi_y": phi_y,
                "dphi_x": dphi_x, "dphi_y": dphi_y,
                "wx": wx, "wy": wy
            }

            f = xs**2
            I = np.sum(wx * f)
            print(I)
            print("CC quadrature of x^2 over [0,1]:", I, " (exact 1/3)")

    def check_orthogonality(self):
        """
        Check the orthogonality and normalization of the cached basis functions.
        """
        if hasattr(self, "_proj_cache"):
            phi_x = self._proj_cache["phi_x"]; phi_y = self._proj_cache["phi_y"]
            wx = self._proj_cache["wx"]; wy = self._proj_cache["wy"]
            Gx = (phi_x * wx) @ phi_x.T
            Gy = (phi_y * wy) @ phi_y.T
            print("‖Gx−I‖∞ =", np.max(np.abs(Gx - np.eye(Gx.shape[0]))))
            print("‖Gy−I‖∞ =", np.max(np.abs(Gy - np.eye(Gy.shape[0]))))

    def check_pde_residual(self):
        """
        Check if cached basis functions approx. solve the 1D eigenvalue ODE
            phi''(x) + k^2 phi(x) = 0
        for each mode k.
        """
        if not hasattr(self, "_proj_cache"):
            print("No cache.")
            return

        xs = self._proj_cache["xs"]
        ys = self._proj_cache["ys"]

        phi_x   = self.phi(xs)        # (M, Nx)
        phi_y   = self.phi(ys)        # (M, Ny)
        ddphi_x = self.ddphi(xs)      # (M, Nx)
        ddphi_y = self.ddphi(ys)      # (M, Ny)

        lam = self.eig_vals**2        # k^2

        # ODE: phi'' + lambda phi = 0
        res_x = ddphi_x + lam[:, None] * phi_x
        res_y = ddphi_y + lam[:, None] * phi_y

        print("‖PDE residual for phi(x)‖∞ =", np.max(np.abs(res_x)))
        print("‖PDE residual for phi(y)‖∞ =", np.max(np.abs(res_y)))

    def check_boundary_residual(self):
        """
        Check if the cached basis functions fulfill the boundary conditions.
        """
        if hasattr(self, "_proj_cache"):
            phi_x = self._proj_cache["phi_x"]; phi_y = self._proj_cache["phi_y"]
            dphi_x = self._proj_cache["dphi_x"]; dphi_y = self._proj_cache["dphi_y"]
            bdry_res_x_0 = self.ibvp.a * phi_x[:, 0] - self.ibvp.b * dphi_x[:, 0]
            bdry_res_x_1 = self.ibvp.a * phi_x[:, -1] + self.ibvp.b * dphi_x[:, -1]
            bdry_res_y_0 = self.ibvp.a * phi_y[:, 0] - self.ibvp.b * dphi_y[:, 0]
            bdry_res_y_1 = self.ibvp.a * phi_y[:, -1] + self.ibvp.b * dphi_y[:, -1]
            print(f"‖Boundary residuals for phi(x)‖∞ = {np.max(np.abs(bdry_res_x_0))}, {np.max(np.abs(bdry_res_x_1))}")
            print(f"‖Boundary residuals for phi(y)‖∞ = {np.max(np.abs(bdry_res_y_0))}, {np.max(np.abs(bdry_res_y_1))}")

    def check_projection(self):
        xs = self._proj_cache["xs"]
        wx = self._proj_cache["wx"]
        phi = self._proj_cache["phi_x"]      # shape (M, n)

        # f = phi_1(x)
        f = phi[1, :]                        # zweite Mode (Index 1)

        # Projektion auf alle Basisfunktionen:
        coeffs = (phi * wx) @ f              # (M,)

        print("coeffs:", coeffs)
        print("max |coeffs|:", np.max(np.abs(coeffs)))
        print("coeff[1]:", coeffs[1])

        xs = self._proj_cache["xs"]
        ys = self._proj_cache["ys"]
        wx = self._proj_cache["wx"]
        wy = self._proj_cache["wy"]
        phi_x = self._proj_cache["phi_x"]
        phi_y = self._proj_cache["phi_y"]

        # phi1(x), phi1(y):
        phi1_x = phi_x[1, :]          # (Nx,)
        phi1_y = phi_y[1, :]          # (Ny,)

        X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
        F = np.outer(phi1_x, phi1_y)  # F(x_i,y_j) = φ1(x_i)φ1(y_j)

        Cf = (phi_x * wx) @ F @ (phi_y * wy).T  # (M,M)

        print("Cf[1,1] ≈", Cf[1,1])
        print("max |Cf[m,n]|, m!=1,n!=1:", np.max(np.abs(Cf - np.eye(Cf.shape[0])[1,1])))

    def check_cached_phis(self):
        """
        Check if the projection cache is prepared.
        """
        self.prepare_cache()
        self.check_orthogonality()
        self.check_pde_residual()
        self.check_boundary_residual()
        self.check_projection()

    def validate_projected_f(self, f_func):
        """
        Validate the projection of the source term f(x,y) onto the modal basis.
        """
        self.prepare_cache()
        xs = self._proj_cache["xs"]; ys = self._proj_cache["ys"]
        wx = self._proj_cache["wx"]; wy = self._proj_cache["wy"]
        phi_x = self._proj_cache["phi_x"]; phi_y = self._proj_cache["phi_y"]
        X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
        F = f_func(X0, Y0)          # preferred signature: f(x,y)
        Cf = (phi_x * wx) @ F @ (phi_y * wy).T # (M, M)
        # print(f"Cf: {Cf}")

        # Reconstruct f from Cf
        F_recon = phi_x.T @ Cf @ phi_y  # (Nx, Ny)
        err = F - F_recon
        print("‖f - f_recon‖_mean ≈", np.mean(err**2)**0.5)

    def calculate_results(self, x, y, t, u0_func, f_func=None, return_u_derivs=True):
        """
        Compute u(x,y,t) and optionally du/dt(x,y,t) for the 2D heat equation
        (inhomogeneous, Robin/Neumann BC) via eigenfunction expansion.

        Parameters
        ----------
        x, y : 1D arrays
            Target coordinates where the solution is evaluated.
        t : float
            Time of evaluation.
        u0_func : callable u0(x,y)
            Initial condition at t=0.
        f_func : callable f(x,y) or f(x,y,t), optional
            (Time-independent) source term. If signature is f(x,y,t), t=0 will be used
            for projection because we assume time-independence in the modal ODEs.
        return_dudt : bool
            If True, return both (U, dUdt). If False, return only U.

        Returns
        -------
        Nothing. Results are stored in self.result_data
        """

        # --- 1) Prepare and cache projection grid and basis evaluations -----------
        self.prepare_cache()

        xs = self._proj_cache["xs"]; ys = self._proj_cache["ys"]
        try:
            dx = self._proj_cache["dx"]; dy = self._proj_cache["dy"]
        except:
            pass
        phi_x = self._proj_cache["phi_x"]; phi_y = self._proj_cache["phi_y"]
        dphi_x = self._proj_cache["dphi_x"]; dphi_y = self._proj_cache["dphi_y"]
        wx = self._proj_cache["wx"]; wy = self._proj_cache["wy"]

        # --- 2) Static offset due to boundary conditions --------------------------
        # For many Robin/Neumann BC setups the stationary solution is constant.
        # We subtract it before projecting and add it back after reconstruction.
        U_ofs = self.ibvp.u_amb()

        # --- 3) Build the 2D spectral operator L = λ_m + λ_n ---------------------
        # Each 2D mode (m,n) decays like exp(-alpha * (λ_m + λ_n) * t).
        k = self.eig_vals
        lam = k**2                                     # shape (M,)
        L = (lam[:, None] + lam[None, :])              # shape (M, M)

        # --- 4) Project initial data onto modal basis -----------------------------
        C0_is_zero= False
        if not hasattr(self, "_C0"):
            X0, Y0 = np.meshgrid(xs, ys, indexing="ij")        # X0: (Nx,Ny), Y0: (Nx,Ny)
            F0 = (u0_func(X0, Y0) - U_ofs)                     # (Nx, Ny)
            if np.allclose(F0, 0.0):
                print("Initial condition is zero everywhere.")
                C0_is_zero= True
                self._C0 = 0.0
            # Matrix multiplications (`@`) are standard NumPy matmul:
            #   phi_x @ F0 @ phi_y.T  ~  ∫∫ phi_m(x) * F0(x,y) * phi_n(y) dx dy
            else:
                print("Projecting initial condition onto modal basis.")
                self._C0 = (phi_x * wx) @ F0 @ (phi_y * wy).T # (M, M)

        # --- 5) Project static source term f(x,y) if present ----------------------
        # We assume f is time-independent for the closed-form solution used here.
        if f_func is not None and not hasattr(self, "_Cf_static"):
            X0, Y0 = np.meshgrid(xs, ys, indexing="ij")
            try:
                F = f_func(X0, Y0)          # preferred signature: f(x,y)
            except TypeError:
                F = f_func(X0, Y0, 0.0)     # fallback if signature is f(x,y,t)
            self._Cf_static = (phi_x * wx) @ F @ (phi_y * wy).T # (M, M)

        # If no source was provided, treat it as zero in the modal equations.
        Cf = getattr(self, "_Cf_static", 0.0)

        # --- 6) Time evolution of modal coefficients C(t) -------------------------
        # Closed-form from the linear ODE for each mode:
        #   C(t) = C0 * exp(-alpha*L*t) + Cf * (1 - exp(-alpha*L*t)) / (alpha*L)
        decay = np.exp(-self.alpha * L * t)                     # (M, M)
        C = self._C0 * decay
        if f_func is not None:
            # small epsilon avoids division by zero for pure zero-modes of L
            C += Cf * (1.0 - decay) / (self.alpha * L + 1e-300)

        # --- 7) Time derivative of the coefficients dC/dt -------------------------
        # Using the ODE directly: dC/dt = -alpha * L * C + Cf
        if return_u_derivs:
            dCdt = -self.alpha * L * C + Cf                     # (M, M)

        # --- 8) Reconstruct on the user-specified evaluation grid -----------------
        # Evaluate basis functions at the requested target points x,y.
        phi_x_eval = self.phi(x) # (M, Nx_eval)
        phi_y_eval = self.phi(y) # (M, Ny_eval)
        dphi_x_eval = self.dphi(x) # (M, Nx_eval)
        dphi_y_eval = self.dphi(y) # (M, Ny_eval)

        # F_proj = phi_x_eval.T @ Cf @ phi_y_eval  # (Nx,Ny)

        #  err = f_func(x,y) - F_proj
        # print("‖f - f_proj‖_mean ≈", np.mean(err**2)**0.5)

        # print(f"Cf: {Cf}")

        # Before reconstruction, we can apply optional filtering to C to
        k = np.arange(len(self.eig_vals))
        kc, p = int(0.85*len(k)), 8
        sigma = np.exp(- (np.maximum(k-kc,0) / max(1, len(k)-kc))**p )
        S = np.diag(sigma)

        # Assemble the fields: U = Uofs + Phi_x^T C Phi_y, dUdt = Phi_x^T dCdt Phi_y
        u = phi_x_eval.T @ C @ phi_y_eval                       # (Nx_eval, Ny_eval)
        if self.use_filter:
            # Now filter C
            C_filt = S @ C @ S
            u = phi_x_eval.T @ C_filt @ phi_y_eval

        u = U_ofs + u
        u_t = phi_x_eval.T @ dCdt @ phi_y_eval
        u_x = dphi_x_eval.T @ C @ phi_y_eval
        u_y = phi_x_eval.T @ C @ dphi_y_eval
        u_xx = (phi_x_eval.T*lam) @ C @ phi_y_eval
        u_yy = phi_x_eval.T @ C @ (phi_y_eval.T*lam).T

        self.result_data = result_data(u, u_t, u_x, u_y, u_xx, u_yy)

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
        X, Y = np.meshgrid(x, y, indexing='xy')
        xy = np.column_stack([X.ravel(), Y.ravel()])
        f = ibvp.heat_source(xy[:,0], xy[:,1])
        f = f.reshape(ny, nx)

        solver = GreenFunctionSolver(ibvp.alpha, ibvp, 1.0, 1.0)

        print("solver.check_cached_phis()")
        solver.check_cached_phis()
        print("solver.validate_projected_f(ibvp.heat_source)")
        solver.validate_projected_f(ibvp.heat_source)

        u_frames = [result_data(ibvp.initial_u(x,y))]

        u0 =ibvp.initial_u(xy[:,0], xy[:,1])
        u0 = u0.reshape(ny, nx)
        u_frames = [result_data(u0)]

        for n_frame in range(n_frames):
            start = time.time()
            tval = frame.lt*(1+n_frame)/n_frames
            solver.calculate_results(x,y,tval,ibvp.initial_u,ibvp.heat_source)
            u_frames.append(solver.result_data)
            u = solver.result_data.u
            min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
            max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
            print(f"Frame {tval:.2f}: mean={u.mean():.6f}, min={u.min():.6f} @ {min_idx}, max={u.max():.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")

        result = result_frames(u_frames, f, has_u_t= True, has_derivs= True, has_laplacian= True)
        return result


