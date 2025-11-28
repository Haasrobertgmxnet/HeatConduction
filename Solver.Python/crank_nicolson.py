import numpy as np
import time
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from boundary_conditions import HeatBoundaryCondition
from result_data import result_data
from result_frames import result_frames

# ----------------------------------------------------
# Debugging and diagnostic helper for matrix structure
# ----------------------------------------------------
def dbg_matrix_checks(solver):
    """
    Perform structural and spectral checks on a Crank-Nicolson solver instance.

    Parameters
    ----------
    solver : HeatCrankNicolsonSolver
        Solver object containing Lh, A, B matrices and grid metadata.

    What it checks
    --------------
    - Magnitude of largest eigenvalues of the B matrix.
    - Sparsity and structure of the discrete Laplacian Lh.
    - Symmetry properties of Lh.
    - Consistency relation: A + B ≈ 2I for theta = 0.5 (Crank-Nicolson).
    - Consistency relation: B - A ≈ dt * alpha * Lh.
    - Tests whether system A x = B * 1 can be solved (detects singularities).
    """
    vals = spla.eigs(solver.B, k=5, which='LM', return_eigenvectors=False)
    print("max |λ(B)| =", np.max(np.abs(vals)))

    k = -1
    j = 1
    r = 0
    print(f"ele at pos {j}, {j}: {solver.Lh[j,j]}")
    for ele in solver.Lh.toarray()[j,:]:
        k = k+1
        if k == j:
            continue
        if ele != 0:
            print(f"ele at pos {j}, {k}: {ele}")
            r =r + np.abs(ele)
        
    print(f"r: {r}")

    # Check eigenvalues of Lh (real parts)
    eig_Lh = spla.eigs(solver.Lh, k=1, which='LR', return_eigenvectors=False)
    print("largest eigenvalue of Lh:", eig_Lh[0])
    eig_Lh = spla.eigs(solver.Lh, k=1, which='SR', return_eigenvectors=False)
    print("smallest eigenvalue of Lh:", eig_Lh[0])

    ny = solver.ny
    nx = solver.nx

    M = solver.Lh.toarray()
    print("M[0:18,0:18] =\n", M[0:18,0:18])

    do_plts = False

    if do_plts:
        # Optional visual diagnostic plotting
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.spy(solver.Lh, markersize=1)
        plt.title("Lh")

        plt.subplot(1,2,2)
        plt.spy(solver.Lh.T - solver.Lh, markersize=1)
        plt.title("Lh.T - Lh (Symmetry check)")
        plt.show()

        plt.figure(figsize=(6,6))
        plt.spy(solver.Lh, markersize=1)
        plt.title("Lh sparsity with block gridlines")

        for j in range(1, ny):
            plt.axhline(j * nx, color='red', lw=0.3, alpha=0.5)
            plt.axvline(j * nx, color='red', lw=0.3, alpha=0.5)
        plt.show()

        plt.figure(figsize=(6,6))
        plt.imshow(solver.Lh.toarray(), cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Matrix value')
        plt.title("Lh (colored by value)")
        for j in range(1, ny):
            plt.axhline(j * (nx+1) - 0.5, color='black', lw=0.3, alpha=0.4)
            plt.axvline(j * (nx+1) - 0.5, color='black', lw=0.3, alpha=0.4)
        plt.show()

    print("A: shape", solver.A.shape, "nnz", solver.A.nnz)
    print("B: shape", solver.B.shape, "nnz", solver.B.nnz)

    for name, M in (("Lh", solver.Lh), ("A", solver.A), ("B", solver.B)):
        vals = M.data if hasattr(M, "data") else M.toarray().ravel()
        print(f"{name}: has_nan={np.isnan(vals).any()}, has_inf={np.isinf(vals).any()}, min={np.nanmin(vals):.3e}, max={np.nanmax(vals):.3e}")

    # Solve A x = B * 1 to detect singularity issues
    try:
        n = solver.A.shape[0]
        test_rhs = solver.B.dot(np.ones(n))
        x = spla.spsolve(solver.A, test_rhs)
        print("spsolve test OK: x min/max", np.min(x), np.max(x))
    except Exception as e:
        print("spsolve Test fehlgeschlagen:", type(e), e)

    # Consistency checks specific to CN time integration
    I = sp.identity(solver.A.shape[0], format='csr')
    test1 = (solver.A + solver.B) - 2 * I
    print("‖A + B - 2I‖ =", np.linalg.norm(test1.toarray(), ord='fro'))

    test2 = (solver.B - solver.A) - solver.dt * solver.alpha * solver.Lh
    print("‖B - A - dt*alpha*Lh‖ =", np.linalg.norm(test2.toarray(), ord='fro'))


# ----------------------------------------------------
# Crank-Nicolson heat equation solver
# ----------------------------------------------------
class HeatCrankNicolsonSolver():
    def __init__(self, alpha, dx, dy, dt, nx, ny, nt, robin):
        """
        Crank-Nicolson implicit solver for the 2D heat equation with Robin boundary conditions.

        Parameters
        ----------
        alpha : float
            Diffusion coefficient.
        dx, dy : float
            Grid spacing in x and y.
        dt : float
            Time step size.
        nx, ny : int
            Number of grid points in x and y directions.
        nt : int
            Total number of time steps (used for pipeline processing).
        robin : tuple (a, b, c)
            Boundary condition parameters for a*u + b*(du/dn) = c.
        """
        self.theta = 0.5  # Crank-Nicolson weighting
        self.alpha = alpha
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.a = robin[0]
        self.b = robin[1]
        self.c = robin[2]

        self.lamx = self.alpha * self.dt / dx**2
        self.lamy = self.alpha * self.dt / dy**2
        self.Lh = None
        self.q = None
        self._factor = None

        self.build_L_h()
        self.crank_nicolson_matrices(self.alpha)

    def build_L_h(self):
        """
        Construct the discrete Laplacian operator Lh using Kronecker products,
        incorporating Robin boundary conditions via ghost-point elimination.

        Returns
        -------
        None
        """
        class D1():
            """
            One-dimensional second derivative matrix with Robin boundary conditions.
            Constructs tri-diagonal operator D and boundary source term q.
            """
            def __init__(self, N, h, a_coef, b_coef, c_coef):
                main = np.zeros(N, dtype=np.float64)
                off1 = np.zeros(N-1, dtype=np.float64)
                off2 = np.zeros(N-1, dtype=np.float64)

                main[1:-1] = -2.0 / h**2
                off1[:] = 1.0 / h**2
                off2[:] = 1.0 / h**2

                q = np.zeros(N, dtype=np.float64)
                tiny = 1e-14

                if abs(b_coef) < tiny:  # Dirichlet
                    if abs(a_coef) < tiny:
                        raise ValueError("Invalid BC: alpha and beta both ~0")
                    main[0] = 0.0
                    main[-1] = 0.0
                    off1[-1] = off2[0] = 0.0
                else:  # Robin
                    main[0] = -2.0 / h**2 - 2.0 * a_coef / (b_coef * h)
                    off2[0] = 2.0 / h**2
                    main[-1] = -2.0 / h**2 - 2.0 * a_coef / (b_coef * h)
                    off1[-1] = 2.0 / h**2
                    q[0] = 2.0 * c_coef / (b_coef * h)
                    q[-1] = 2.0 * c_coef / (b_coef * h)

                self.q = q
                self.D = sp.diags([off1, main, off2], offsets=[-1,0,1], shape=(N, N), format='csr')

        DxInstance = D1(self.nx, self.dx, self.a, self.b, self.c)
        DyInstance = D1(self.ny, self.dy, self.a, self.b, self.c)

        Dx = DxInstance.D
        Dy = DyInstance.D
        qx = DxInstance.q
        qy = DyInstance.q

        Ix = sp.identity(self.nx)
        Iy = sp.identity(self.ny)
        self.Lh = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)

        qx_2d = np.kron(np.ones(self.ny), qx)
        qy_2d = np.kron(qy, np.ones(self.nx))

        self.q_total = qx_2d + qy_2d

        corner_indices = [
            0,
            self.nx - 1,
            (self.ny - 1) * self.nx,
            self.ny * self.nx - 1
        ]
        #for idx in corner_indices:
        #    self.q_total[idx] = 0.5 * (qx_2d[idx] + qy_2d[idx])

    def crank_nicolson_matrices(self, kappa):
        """
        Construct A and B matrices for Crank-Nicolson update:
            A u^{n+1} = B u^{n} + dt*(q_total * alpha + f)

        Parameters
        ----------
        kappa : float
            Diffusion coefficient (often equals alpha).
        """
        print(f"Kappa: {kappa}")
        n = self.Lh.shape[0]
        I = sp.identity(n, format='csc')
        Lh_csr = self.Lh.tocsr()
        A = (I - (1 - self.theta) * self.dt * kappa * Lh_csr)
        B = (I + self.theta * self.dt * kappa * Lh_csr)
        self.A = A.tocsc()
        self.B = B.tocsr()

        try:
            self._factor = spla.factorized(self.A)
        except Exception:
            self._splu = spla.splu(self.A)
            self._factor = lambda rhs: self._splu.solve(rhs)

    def check_stability(self):
        """
        Return stability indicator (for reference only; CN is unconditionally stable).
        """
        stability_number = self.lamx + self.lamy
        return stability_number <= 0.5, stability_number

    def step(self, u, f = None):
        """
        Perform one Crank-Nicolson time step.

        Parameters
        ----------
        u : ndarray (ny, nx) or 1D flattened version
            Current temperature field.
        f : ndarray (ny, nx), optional
            Source term.

        Returns
        -------
        u_new : ndarray
            Updated field, same shape as input u.
        """
        input_was_2d = (u.ndim == 2)
        if input_was_2d:
            u_vec = u.ravel(order='C')
        else:
            u_vec = u.copy()

        if f is None:
            f_vec = np.zeros_like(u_vec)
        else:
            if f.ndim == 2:
                f_vec = f.ravel(order='C')
            else:
                f_vec = f

        rhs = self.B.dot(u_vec) + self.dt * f_vec + self.dt * self.q_total * self.alpha

        if self._factor is not None and self.theta < 1.0:
            u_new_vec = self._factor(rhs)
        if self._factor is None and self.theta < 1.0:
            u_new_vec = spla.spsolve(self.A, rhs)
        if self.theta == 1.0:
            u_new_vec = 1.0*rhs

        if input_was_2d:
            u_new = u_new_vec.reshape((self.ny, self.nx), order='C')
            return u_new
        else:
            return u_new_vec

    def n_steps(self, u, f = None, nt=1):
        """
        Perform nt Crank-Nicolson time steps.
        """
        u_t = None
        for _ in range(nt):
            u_old = u
            u = self.step(u, f)
            u_t = (u - u_old) / self.dt
        self.current_u = u
        self.current_u_t = u_t
        return u, u_t

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
        """
        Run the solver over multiple frames in time and collect solution snapshots.

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
        print("Crank-Nicolson solver")
        nx, ny = frame.nx, frame.ny
        lx, ly = frame.lx, frame.ly
        lt = frame.lt
        dt = lt / frame.nt

        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y, indexing='xy')
        X2, Y2 = np.meshgrid(x, y, indexing='xy')
        u0 = ibvp.initial_u(X2.ravel(), Y2.ravel()).reshape((ny, nx), order='C')
        f = ibvp.heat_source(X2.ravel(), Y2.ravel()).reshape((ny, nx), order='C')
        robin =  HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c).to_tuple_x()

        dx, dy = lx / (nx-1), ly / (ny-1)
        solver = HeatCrankNicolsonSolver(ibvp.alpha, dx, dy, dt, nx, ny, frame.nt, robin)
        stable, sn = solver.check_stability()
        if not stable:
            print("CFL condition (lamx+lamy) = {:.4g} > 0.5".format(sn))

        u_corr = solver.c / solver.a
        u_corr = 0
        u_frames = [result_data(u0.copy())]
        u = u0.copy()

        for n_frame in range(n_frames):
            start = time.time()
            u, u_t = solver.n_steps(u, f, t_steps_per_frame)
            u_frames.append(result_data(u, u_t))

            min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
            max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
            tval = (n_frame + 1) * (lt / n_frames)
            print(f"Frame {tval:.2f}: mean={u.mean():.6f}, min={u.min():.6f} @ {min_idx}, max={u.max():.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")
        
        result = result_frames(u_frames, f, has_u_t= True, has_derivs= False, has_laplacian= False)
        return result
