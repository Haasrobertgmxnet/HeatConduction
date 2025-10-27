import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from boundary_conditions import HeatBoundaryCondition

# 1) schnelle Matrix-/RHS-Checks
def dbg_matrix_checks(solver):
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

    Dx, qx_left = build_D1(solver.nx, solver.dx, *solver.robin_x)
    M = Dx.toarray()
    print("Dx[0:6, 0:8] =\n", M[0:6, 0:8])

    eig_Dx = spla.eigs(Dx, k=1, which='LR', return_eigenvectors=False)
    print("largest eigenvalue of Dx:", eig_Dx[0])
    eig_Dx = spla.eigs(Dx, k=1, which='SR', return_eigenvectors=False)
    print("smallest eigenvalue of Dx:", eig_Dx[0])

    Dy, qy_left = build_D1(solver.ny, solver.dy, *solver.robin_y)
    M = Dy.toarray()
    print("Dy[0:6, 0:8] =\n", M[0:6, 0:8])

    eig_Dy = spla.eigs(Dy, k=1, which='LR', return_eigenvectors=False)
    print("largest eigenvalue of Dy:", eig_Dy[0])
    eig_Dy = spla.eigs(Dy, k=1, which='SR', return_eigenvectors=False)
    print("smallest eigenvalue of Dy:", eig_Dy[0])

    ### Check Lh

    # 1) größter reeller Eigenwert von Lh (LR = largest real part)
    eig_Lh = spla.eigs(solver.Lh, k=1, which='LR', return_eigenvectors=False)
    print("largest eigenvalue of Lh:", eig_Lh[0])
    eig_Lh = spla.eigs(solver.Lh, k=1, which='SR', return_eigenvectors=False)
    print("smallest eigenvalue of Lh:", eig_Lh[0])

    # 2) kleinster (most negative) Eigenwert
    ny = solver.ny  # oder nx, je nachdem
    nx = solver.nx

    # 3)
    # kleine Ansicht der ersten 6x8 Einträge
    M = solver.Lh.toarray()
    print("Lh[0:6, 0:8] =\n", M[0:6, 0:8])
    # direkte Abfrage der fraglichen Einträge:
    print("Lh[0,1] =", M[0,1])
    print("Lh[1,2] =", M[1,2])

    # 4)

    do_plts = False

    if do_plts:
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

        # Gitterlinien auf Blockgrenzen zeichnen:
        for j in range(1, ny):
            plt.axhline(j * nx, color='red', lw=0.3, alpha=0.5)
            plt.axvline(j * nx, color='red', lw=0.3, alpha=0.5)

        plt.show()

        plt.figure(figsize=(6,6))
        plt.imshow(solver.Lh.toarray(), cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Matrix value')
        plt.title("Lh (colored by value)")

        nx, ny = solver.nx, solver.ny
        for j in range(1, ny):
            plt.axhline(j * (nx+1) - 0.5, color='black', lw=0.3, alpha=0.4)
            plt.axvline(j * (nx+1) - 0.5, color='black', lw=0.3, alpha=0.4)

        plt.show()

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

    # Test 1: A + B ≈ 2I
    I = sp.identity(solver.A.shape[0], format='csr')
    test1 = (solver.A + solver.B) - 2 * I
    print("‖A + B - 2I‖ =", np.linalg.norm(test1.toarray(), ord='fro'))

    # Test 2: B - A ≈ dt * alpha * Lh
    test2 = (solver.B - solver.A) - solver.dt * solver.alpha * solver.Lh
    print("‖B - A - dt*alpha*Lh‖ =", np.linalg.norm(test2.toarray(), ord='fro'))

def build_D1(N, h, alpha_left, beta_left, g_left,
             alpha_right, beta_right, g_right):
    # beta_left = -beta_left

    main = np.zeros(N, dtype=np.float64)
    off1 = np.zeros(N-1, dtype=np.float64)
    off2 = np.zeros(N-1, dtype=np.float64)

    # Innenpunkte (i=1..N-1)
    main[1:-1] = -2.0 / h**2
    off1[:] = 1.0 / h**2
    off2[:] = 1.0 / h**2

    q = np.zeros(N, dtype=np.float64)

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
        main[0] = -2.0 / h**2 - 2.0 * alpha_left / (beta_left * h)
        off2[0] = 2.0 / h**2
        q[0]    = -2.0 * g_left / (beta_left * h)

    # --- RIGHT boundary (i=N) ---
    if abs(beta_right) < tiny:  # Dirichlet at right
        if abs(alpha_right) < tiny:
            raise ValueError("Ungültige rechte BC: alpha_right und beta_right beide ~0")
        main[-1] = 1.0
        off1[-1] = off2[-1] = 0.0
        q[-1]    = g_right / alpha_right
    else:
        main[-1] = -2.0 / h**2 - 2.0 * alpha_right / (beta_right * h)
        # off2[-1] = 1.0 / h**2
        off1[-1] = 2.0 / h**2
        q[-1]    = -2.0 * g_right / (beta_right * h)

    D = sp.diags([off1, main, off2], offsets=[-1,0,1], shape=(N, N), format='csr')
    return D, q

def build_D1_fixed(npts, h, alpha_left, beta_left, g_left,
                   alpha_right, beta_right, g_right):
    """
    Baut 1D Laplace-Operator mit Robin-Randbedingungen.
    
    Args:
        npts: Anzahl der Gitterpunkte (nx oder ny)
        h: Gitterabstand dx oder dy
        alpha_left, beta_left, g_left: Linke RB: alpha*u + beta*du/dn = g
        alpha_right, beta_right, g_right: Rechte RB
    
    Returns:
        D: Sparse Matrix (npts x npts)
        q: RHS-Vektor für Randbedingungen
    """
    tiny = 1e-14
    
    main = np.zeros(npts)
    lower = np.zeros(npts-1)  # untere Nebendiagonale (offset=-1)
    upper = np.zeros(npts-1)  # obere Nebendiagonale (offset=+1)
    q = np.zeros(npts)
    
    # Innere Punkte: Standard 3-Punkt-Stencil
    main[1:-1] = -2.0 / h**2
    lower[:] = 1.0 / h**2   # Verbindet i mit i-1
    upper[:] = 1.0 / h**2   # Verbindet i mit i+1
    
    # --- LINKER RAND (i=0) ---
    if abs(beta_left) < tiny:
        # Dirichlet: alpha*u[0] = g_left
        if abs(alpha_left) < tiny:
            raise ValueError("Ungültige linke RB: alpha und beta beide ~0")
        main[0] = 1.0
        upper[0] = 0.0
        q[0] = g_left / alpha_left
    else:
        # Robin: alpha*u[0] + beta*(u[1]-u[-1])/(2h) = g_left
        # Ghost-elimination: u[-1] = u[1] - 2h*g/beta + 2h*alpha*u[0]/beta
        # u''[0] = (u[-1] - 2*u[0] + u[1])/h²
        #        = (2*u[1] - 2*u[0])/h² + 2*alpha*u[0]/(beta*h) - 2*g/(beta*h)
        main[0] = -2.0 / h**2 - 2.0 * alpha_left / (beta_left * h)
        upper[0] = 2.0 / h**2
        q[0] = -2.0 * g_left / (beta_left * h)
    
    # --- RECHTER RAND (i=npts-1) ---
    if abs(beta_right) < tiny:
        # Dirichlet
        if abs(alpha_right) < tiny:
            raise ValueError("Ungültige rechte RB: alpha und beta beide ~0")
        main[-1] = 1.0
        lower[-1] = 0.0
        q[-1] = g_right / alpha_right
    else:
        # Robin: alpha*u[N] + beta*(u[N+1]-u[N-1])/(2h) = g_right
        # u[N+1] = u[N-1] + 2h*g/beta - 2h*alpha*u[N]/beta
        # u''[N] = (u[N-1] - 2*u[N] + u[N+1])/h²
        #        = (2*u[N-1] - 2*u[N])/h² - 2*alpha*u[N]/(beta*h) + 2*g/(beta*h)
        main[-1] = -2.0 / h**2 + 2.0 * alpha_right / (beta_right * h)
        lower[-1] = 2.0 / h**2
        q[-1] = -2.0 * g_right / (beta_right * h)
    
    # Sparse Matrix: lower (offset=-1), main (offset=0), upper (offset=+1)
    D = sp.diags(
        [lower, main, upper],
        offsets=[-1, 0, 1],
        shape=(npts, npts),
        format='csr'
    )
    
    return D, q


# Die build_L_h Funktion bleibt unverändert (war bereits korrekt für nx×ny Konvention)
def build_L_h_fixed(nx, ny, dx, dy, robin_x, robin_y):
    """
    Baut 2D Laplace-Operator für nx×ny Gitter.
    
    Args:
        nx, ny: Anzahl Gitterpunkte (NICHT Intervalle!)
        dx, dy: Gitterabstände
        robin_x: (alpha_left, beta_left, g_left, alpha_right, beta_right, g_right)
        robin_y: (alpha_bottom, beta_bottom, g_bottom, alpha_top, beta_top, g_top)
    """
    # nx, ny sind bereits Anzahl Punkte
    Dx, qx = build_D1(nx, dx, *robin_x)
    Dy, qy = build_D1(ny, dy, *robin_y)
    
    Ix = sp.identity(nx, format='csr')
    Iy = sp.identity(ny, format='csr')
    
    # 2D Laplace: Lh = Dy ⊗ Ix + Iy ⊗ Dx
    Lh = sp.kron(Dy, Ix, format='csr') + sp.kron(Iy, Dx, format='csr')
    
    # RHS für Randbedingungen
    qx_2d = np.kron(np.ones(ny), qx)
    qy_2d = np.kron(qy, np.ones(nx))
    q_total = qx_2d + qy_2d
    
    return Lh, q_total

# ---------- Korrigierte Solver-Klasse ----------
class HeatCrankNicolsonSolver():
    def __init__(self, alpha, dx, dy, dt, nx, ny, nt, robin_x, robin_y):
        """
        nx, ny: Anzahl Intervalle (nicht Anzahl Punkte). Punkte sind nx+1, ny+1.
        """
        self.theta = 0.5
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
        #self.Lh, self.q_total = build_L_h_fixed(self.nx, self.ny, self.dx, self.dy, self.robin_x, self.robin_y)
        #return
        print(f"self.dx: {self.dx}")
        print(f"self.nx: {self.nx}")
        print(f"self.dx * self.nx: {self.dx*(self.nx-1)}")
        Dx, qx_left = build_D1(self.nx, self.dx, *self.robin_x)
        Dy, qy_bottom = build_D1(self.ny, self.dy, *self.robin_y)
        Ix = sp.identity(self.nx)
        Iy = sp.identity(self.ny)
        Ireg = -3e3*sp.identity(self.nx*self.ny)
        self.Lh = sp.kron(Iy, Dx) + sp.kron(Dy, Ix) # + Ireg
        # Rand-RHS
        qx_2d = np.kron(np.ones(self.ny), qx_left)
        qy_2d = np.kron(qy_bottom, np.ones(self.nx))
        self.q_total = qx_2d + qy_2d   # Länge (nx+1)*(ny+1)

    def crank_nicolson_matrices(self, kappa):
        print(f"Kappa: {kappa}")
        n = self.Lh.shape[0]
        # Use csc for SuperLU
        I = sp.identity(n, format='csc')
        Lh_csr = self.Lh.tocsr()
        A = (I - (1 - self.theta) * self.dt * kappa * Lh_csr)
        B = (I + self.theta * self.dt * kappa * Lh_csr)
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
        # rhs = self.B.dot(u_vec) + self.dt * (f_vec + self.q_total)
        # rhs = p2 + p3 - 0.11*p4
        rhs = p2 + p3 + p4
        #print(f"RHS MAX: {np.mean(rhs)}")
        # print(f"rhs max: {rhs.max()}")

        # print(f"Shape: {u_vec.shape}")

        # Löse (verwende faktorisierten Solver falls vorhanden)
        if self._factor is not None:
            u_new_vec = self._factor(rhs)
        else:
            u_new_vec = spla.spsolve(self.A, rhs)

        #print(f"Means: u old: {np.mean(u_vec)}, u new: {np.mean(u_new_vec)}")
        #print(f"Maxs: u old: {np.max(u_vec)}, u new: {np.max(u_new_vec)}")
        if input_was_2d:
            u_new = u_new_vec.reshape((self.ny, self.nx), order='C')
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
        # nxp, nyp = nx+1, ny+1
        dt = lt / frame.nt

        # Gitterpunkte
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
        X, Y = np.meshgrid(x, y, indexing='xy')   # shape (nyp, nxp) if indexing='xy' -> careful
        # Besser für unsere ravel(order='C') Ziel: shape (ny+1, nx+1) with row index = y
        X2, Y2 = np.meshgrid(x, y, indexing='xy')
        # initial u: ensure shape (ny+1, nx+1)
        u0 = ibvp.initial_u(X2.ravel(), Y2.ravel()).reshape((ny, nx), order='C')
        h = ibvp.heat_source(X2.ravel(), Y2.ravel()).reshape((ny, nx), order='C')

        neumann_bc = HeatBoundaryCondition(ibvp.a, ibvp.b, ibvp.c)
        dx, dy = lx / (nx-1), ly / (ny-1)

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
            print(f"Frame {tval:.2f}: u mean={u_mean:.6f}, ")
            # print(f"NC *** Frame {n_frame+1}/{n_frames} t={tval:.4f}: mean u = {u_mean:.6e}")
        return frames, u_means
