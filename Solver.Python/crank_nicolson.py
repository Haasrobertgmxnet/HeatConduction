import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from boundary_conditions import HeatBoundaryCondition

# 1) schnelle Matrix-/RHS-Checks
def dbg_matrix_checks(solver):
    vals = spla.eigs(solver.B, k=5, which='LM', return_eigenvectors=False)
    print("max |λ(B)| =", np.max(np.abs(vals)))

    # 1) größter reeller Eigenwert von Lh (LR = largest real part)
    eig_Lh = spla.eigs(solver.Lh, k=1, which='LR', return_eigenvectors=False)
    print("largest eigenvalue of Lh:", eig_Lh[0])

    # 2) kleinster (most negative) Eigenwert
    eig_Lh_min = spla.eigs(solver.Lh, k=1, which='SR', return_eigenvectors=False)
    print("smallest eigenvalue of Lh:", eig_Lh_min[0])
    print("Lh: shape", solver.Lh.shape, "nnz", solver.Lh.nnz)
    print("A: shape", solver.A.shape, "nnz", solver.A.nnz)
    print("B: shape", solver.B.shape, "nnz", solver.B.nnz)
    # prüfen auf NaN/Inf in Datenarrays
    for name, M in (("Lh", solver.Lh), ("A", solver.A), ("B", solver.B)):
        vals = M.data if hasattr(M, "data") else M.toarray().ravel()
        print(f"{name}: has_nan={np.isnan(vals).any()}, has_inf={np.isinf(vals).any()}, min={np.nanmin(vals):.3e}, max={np.nanmax(vals):.3e}")
    # q_total
    q = solver.q_total
    print("q_total: has_nan", np.isnan(q).any(), "has_inf", np.isinf(q).any(), "min", np.nanmin(q), "max", np.nanmax(q))
    # Testlösung: solve A x = B*(const vector) — prüft Singularität
    try:
        n = solver.A.shape[0]
        test_rhs = solver.B.dot(np.ones(n))
        x = spla.spsolve(solver.A, test_rhs)
        print("spsolve test OK: x min/max", np.min(x), np.max(x))
    except Exception as e:
        print("spsolve Test fehlgeschlagen:", type(e), e)

def build_D1(N, h, alpha_left, beta_left, g_left,
             alpha_right, beta_right, g_right):
    beta_left = -beta_left

    npts = N + 1
    main = np.zeros(npts, dtype=np.float64)
    off1  = off2 = np.zeros(npts-1, dtype=np.float64)

    # Innenpunkte (i=1..N-1)
    main[1:-1] = -2.0 / h**2
    off1[1:-1] = off2[1:-1] = 1.0 / h**2

    q = np.zeros(npts, dtype=np.float64)

    tiny = 1e-14

    # --- LEFT boundary (i=0) ---
    if abs(beta_left) < tiny:  # Dirichlet: alpha*u0 = g_left  => u0 = g_left/alpha
        if abs(alpha_left) < tiny:
            raise ValueError("Ungültige linke BC: alpha_left und beta_left beide ~0")
        main[0] = 1.0
        off1[0] = off2[0] = 0.0
        q[0]    = g_left / alpha_left
    else:
        # Ghost-elimination (siehe Ableitung)
        main[0] = -2.0 / h**2 + 2.0 * alpha_left / (beta_left * h)
        off1[0] = main[0]
        off2[0]  = 1.0 / h**2
        q[0]    = -2.0 * g_left / (beta_left * h)

    # --- RIGHT boundary (i=N) ---
    if abs(beta_right) < tiny:  # Dirichlet at right
        if abs(alpha_right) < tiny:
            raise ValueError("Ungültige rechte BC: alpha_right und beta_right beide ~0")
        main[-1] = 1.0
        off1[-1] = off2[-1] = 0.0
        q[-1]    = g_right / alpha_right
    else:
        main[-1] = -2.0 / h**2 + 2.0 * alpha_right / (beta_right * h)
        off2[-1] = main[-1]
        off1[-1]  = 1.0 / h**2
        q[-1]    = -2.0 * g_right / (beta_right * h)

    D = sp.diags([off1, main, off2], offsets=[-1,0,1], shape=(npts, npts), format='csr')
    return D, q

# --- (build_D1, build_L_h, crank_nicolson_matrices unverändert, außer kleine Anpassungen) ---
# Ich nehme hier an, diese Funktionen sind identisch zu Ihrer bisherigen Implementierung,
# nur dass "Nx" bedeutet: Anzahl Intervalle => Punkte = Nx+1.

# ---------- Korrigierte Solver-Klasse ----------
class HeatCrankNicolsonSolver():
    def __init__(self, alpha, dx, dy, dt, nx, ny, nt, robin_x, robin_y):
        """
        nx, ny: Anzahl Intervalle (nicht Anzahl Punkte). Punkte sind nx+1, ny+1.
        """
        self.mu = 0.0
        self.alpha = alpha
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.robin_x = robin_x
        self.robin_y = robin_y

        # Stabilitätskennzahl (nur informativ für CN)
        self.lamx = self.alpha * self.dt / dx**2
        self.lamy = self.alpha * self.dt / dy**2
        self.Lh = None
        self._factor = None

        # Aufbau Matrizen
        self.build_L_h()
        self.crank_nicolson_matrices(self.alpha)

        dbg_matrix_checks(self)

    def build_L_h(self):
        Dx, qx_left = build_D1(self.nx, self.dx, *self.robin_x)
        Dy, qy_bottom = build_D1(self.ny, self.dy, *self.robin_y)
        Ix = sp.identity(self.nx+1)
        Iy = sp.identity(self.ny+1)
        self.Lh = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)
        # Rand-RHS
        qx_2d = np.kron(np.ones(self.ny+1), qx_left)
        qy_2d = np.kron(qy_bottom, np.ones(self.nx+1))
        self.q_total = qx_2d + qy_2d   # Länge (nx+1)*(ny+1)

    def crank_nicolson_matrices(self, kappa):
        n = self.Lh.shape[0]
        # Use csc for SuperLU
        I = sp.identity(n, format='csc')
        Lh_csr = self.Lh.tocsr()
        A = (I - (1 - self.mu) * self.dt * kappa * Lh_csr)
        B = (I + self.mu * self.dt * kappa * Lh_csr)
        # keep them in sparse formats you need:
        self.A = A.tocsc()    # important: factorization likes csc
        self.B = B.tocsr()

        # Factorize once with SuperLU (fast apply later)
        # spla.factorized expects csc/CSR (returns function calling SuperLU)
        try:
            fa = None
            fa = spla.factorized(self.A) 
            self._factor = spla.factorized(self.A)  # returns function rhs -> x
        except Exception:
            # fallback: precompute splu object and wrap it
            self._splu = spla.splu(self.A)
            self._factor = lambda rhs: self._splu.solve(rhs)

    def check_stability(self):
        stability_number = self.lamx + self.lamy
        return stability_number <= 0.5, stability_number

    def step(self, u, f = None):
        # Akzeptiere sowohl 2D als auch 1D u:
        
        input_was_2d = (u.ndim == 2)
        if input_was_2d:
            # In Ihrer Kronecker-Bauweise ist die Ordnung lexicographisch
            # (x variiert schneller). Wir benutzen 'C' order (row-major) konsistent.
            u_vec = u.ravel(order='C')   # Länge (nx+1)*(ny+1)
        else:
            u_vec = u.copy()

        if f is None:
            f_vec = np.zeros_like(u_vec)
        else:
            if f.ndim == 2:
                f_vec = f.ravel(order='C')
            else:
                f_vec = f

        # print(f"self.dt: { self.dt }")
        p0 = u_vec
        p1 = 25.0*self.B.dot(np.ones_like(u_vec))
        p2 = self.B.dot(u_vec)
        p3 = self.dt * f_vec
        p4 = self.dt * self.q_total

        #for i, p in enumerate([p0, p1, p2, p3, p4], start=1):
            #print(f"Part {i} max: {np.max(p):.6e}, mean: {np.mean(p):.6e}")

        # if f_vec is old and new
        # rhs = self.B.dot(u_vec) + 0.5*self.dt*(f_vec + f_vec) + 0.5*self.dt*(self.q_total + self.q_total)
        rhs = self.B.dot(u_vec) + self.dt * (f_vec + self.q_total)
        # rhs = p0 + p3 + p4
        #print(f"RHS MAX: {np.mean(rhs)}")
        # print(f"rhs max: {rhs.max()}")

        print(f"Shape: {u_vec.shape}")

        # Löse (verwende faktorisierten Solver falls vorhanden)
        if self._factor is not None:
            u_new_vec = self._factor(rhs)
        else:
            u_new_vec = spla.spsolve(self.A, rhs)

        #print(f"Means: u old: {np.mean(u_vec)}, u new: {np.mean(u_new_vec)}")
        #print(f"Maxs: u old: {np.max(u_vec)}, u new: {np.max(u_new_vec)}")
        if input_was_2d:
            u_new = u_new_vec.reshape((self.ny+1, self.nx+1), order='C')
            return u_new
        else:
            return u_new_vec

    def n_steps(self, u, f = None, nt=1):
        for _ in range(nt):
            u = self.step(u, f)
        return u

    # --------- Korrigierte pipeline-Funktion (als freie Funktion) ----------
    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
        nx, ny = frame.nx, frame.ny
        lx, ly = frame.lx, frame.ly
        lt = frame.lt

        # Anzahl Punkte:
        nxp, nyp = nx+1, ny+1
        dt = lt / frame.nt

        # Gitterpunkte
        x = np.linspace(0, lx, nxp)
        y = np.linspace(0, ly, nyp)
        X, Y = np.meshgrid(x, y, indexing='xy')   # shape (nyp, nxp) if indexing='xy' -> careful
        # Besser für unsere ravel(order='C') Ziel: shape (ny+1, nx+1) with row index = y
        X2, Y2 = np.meshgrid(x, y, indexing='xy')
        # initial u: ensure shape (ny+1, nx+1)
        u0 = ibvp.initial_u(X2.ravel(), Y2.ravel()).reshape((ny+1, nx+1), order='C')
        h = ibvp.heat_source(X2.ravel(), Y2.ravel()).reshape((ny+1, nx+1), order='C')

        neumann_bc = HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c)
        dx, dy = lx / nx, ly / ny

        robin_x = neumann_bc.to_tuple_x()
        robin_y = neumann_bc.to_tuple_y()

        solver = HeatCrankNicolsonSolver(ibvp.alpha, dx, dy, dt, nx, ny, frame.nt, robin_x, robin_y)
        stable, sn = solver.check_stability()
        if not stable:
            print("CFL condition (lamx+lamy) = {:.4g} > 0.5".format(sn))

        frames = [u0.copy()]
        u = u0.copy()
        u_means = []
        for n_frame in range(n_frames):
            u = solver.n_steps(u, h, t_steps_per_frame)
            frames.append(u.copy())
            u_mean = u.mean()
            u_means.append(u_mean)
            tval = (n_frame+1) * (lt / n_frames)
            print(f"NC *** Frame {n_frame+1}/{n_frames} t={tval:.4f}: mean u = {u_mean:.6e}")
        return frames, u_means
