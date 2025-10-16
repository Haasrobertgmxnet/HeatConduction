import numpy as np
from boundary_conditions import HeatBoundaryCondition
import time

class GreenFunctionSolver:
    def __init__(self, alpha, bc, Lx=1.0, Ly=1.0, M=20, N=20):
        print(f"Gesuchter Wert: {0.48503638*204/214}")
        self.alpha = alpha
        self.apply_bc = bc

        self.eig_vals = np.array([ 0.96018887, 3.43101431, 6.43819715, 9.52961783, 12.64540952,
         15.77134816, 18.90244679, 22.03652001, 25.17246196, 28.30965385, 31.44772266,
         34.58643025, 37.72561748, 40.86517399, 44.00502085, 47.14510012, 50.2853683,
         53.42579212, 56.56634566])

        self.scal = np.array([0.4787, 0.1087893,  0.04307502, 0.0239897, 0.01570866,
         0.01128258, 0.00860051, 0.00683343, 0.00559754, 0.0046936, 0.00400903,
         0.00347597, 0.00305132, 0.00270657, 0.00242215, 0.00218426, 0.00198289,
         0.00181066, 0.00166199])



    def phi(self, eig_vals, x):
        mode = np.atleast_1d(eig_vals)[:, None]   # (M,1)
        x = np.atleast_1d(x)[None, :]   # (1,N)
        gamma = self.apply_bc.a / self.apply_bc.b
        phi_vals = np.sin(mode * x) + (mode/gamma)*np.cos(mode * x)

        debug_ = False

        if not debug_:
            return phi_vals

        if debug_:
            arg = mode * x
            print(f"Size x {x.size}")
            print(f"Size mode {mode.size}")
            print(f"Size arg {arg.size}")
            print(f"Size phi: {phi_vals.size}")

        return phi_vals

    # vereinfachte, korrigierte green-Funktion (konzeptuell)
    def green(self, x,y,x0,y0,tau):
        if tau < 0:
            return np.zeros((len(x), len(y)))

        eig_vals = self.eig_vals
        scal = self.scal
        
        k = np.atleast_1d(eig_vals)
        # numerische Normierung auf ganzen Gitterpunkten xs, ys
        xs = ys = np.linspace(0.,1.,20)
        PHI_x = self.phi(k, xs)   # shape (M, Nx)
        PHI_y = self.phi(k, ys)   # shape (M, Ny)
        dx = np.mean(np.diff(xs)); dy = np.mean(np.diff(ys))

        
        variant1 = False
        if variant1:
            norms_x = np.sqrt(np.sum(PHI_x**2, axis=1)*dx)
            norms_y = np.sqrt(np.sum(PHI_y**2, axis=1)*dy)
            scal = 1/norms_x

        PHI_x_norm = self.phi(k, x)*scal[:,None]
        PHI_x0_norm = self.phi(k, x0)*scal[:,None]
        PHI_y_norm = self.phi(k, y)*scal[:,None]
        PHI_y0_norm = self.phi(k, y0)*scal[:,None]
        # PHI_x_norm = self.phi(k, x) / norms_x[:,None]
        # PHI_x0_norm = self.phi(k, x0) / norms_x[:,None]
        # PHI_y_norm = self.phi(k, y) / norms_y[:,None]
        # PHI_y0_norm = self.phi(k, y0) / norms_y[:,None]

        km2 = k**2

        ex = self.alpha * (km2[:,None] + km2[None,:])
        A = np.exp(-ex * tau)   # shape (M,M)
        A_int = 1/ex*(1-A)

        # Vectorized assemble (fast): build C = A @ (psi_y0 * psi_y)
        C = A @ (PHI_y0_norm[:,0][:,None] * PHI_y_norm)  # (M, Ny)
        D = (PHI_x0_norm[:,0][:,None] * PHI_x_norm)      # (M, Nx)
        G = D.T @ C   # (Nx, Ny)

        C_int = A_int @ (PHI_y0_norm[:,0][:,None] * PHI_y_norm)  # (M, Ny)
        D = (PHI_x0_norm[:,0][:,None] * PHI_x_norm)      # (M, Nx)
        G_int = D.T @ C_int   # (Nx, Ny)

        return G, G_int


    def green_old(self, x, y, x0, y0, tau):
        if tau < 0.0:
            return 0.0

        eig_vals = self.eig_vals19

        scal = np.array([0.48503638, 0.1087893,  0.04307502, 0.0239897, 0.01570866,
         0.01128258, 0.00860051, 0.00683343, 0.00559754, 0.0046936, 0.00400903,
         0.00347597, 0.00305132, 0.00270657, 0.00242215, 0.00218426, 0.00198289,
         0.00181066, 0.00166199])

        eig_vals = self.eig_vals10

        scal = np.array([0.48503638, 0.1087893,  0.04307502, 0.0239897, 0.01570866,
         0.01128258, 0.00860051, 0.00683343, 0.00559754, 0.0046936])

        #eig_vals = np.array([ 0.96018887, 3.43101431, 6.43819715, 9.52961783, 12.64540952])

        #scal = np.array([0.48503638, 0.1087893,  0.04307502, 0.0239897, 0.01570866])

        # A = np.exp(-self.alpha * 2*eig_vals[None, :]**2 * tau)
        A0 = np.exp(-self.alpha * 0.5 * eig_vals[None, :]**2 * tau)

        dx = np.mean(np.diff(x))
        t = self.phi(eig_vals,  np.atleast_1d(x))
        s = np.sum(t**2, axis=1)*dx
        s = 1/np.sqrt(s)

        # scal = scal[:, None] * A0.T
        scal = s[:, None] * A0.T
        phi_m_x  = scal*self.phi(eig_vals,  np.atleast_1d(x))    # (M, Nx)
        phi_m_x0 = scal*self.phi(eig_vals,  np.atleast_1d(x0))   # (M, N0)
        phi_n_y  = scal*self.phi(eig_vals,  np.atleast_1d(y))    # (N, Ny)
        phi_n_y0 = scal*self.phi(eig_vals,  np.atleast_1d(y0))   # (N, N0)

        G = np.einsum('ip,jq->pq', phi_m_x * phi_m_x0, phi_n_y * phi_n_y0)
        return G

        term1 = phi_m_x * phi_m_x0
        term2 = phi_n_y * phi_n_y0

        print(f"phi_m_x.shape: {phi_m_x.shape}")
        print(f"phi_m_x0.shape: {phi_m_x0.shape}")
        print(f"phi_n_y.shape: {phi_n_y.shape}")
        print(f"phi_n_y0.shape: {phi_n_y0.shape}")

        print(f"term1.shape: {term1.shape}")
        print(f"term2.shape: {term2.shape}")

        print(f"G.shape: {G.shape}")
        return G

    def u(self, x, y, t, u0_func, f_func=None):
        green = self.green
        nx, ny = 101, 101  # ungerade!
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)

        U_ofs = self.apply_bc.c / self.apply_bc.a

        G = np.zeros((nx, ny, len(x), len(y)))
        for i, x0 in enumerate(xs):
            for j, y0 in enumerate(ys):
                g1, g2 = green(x, y, x0, y0, t)
                phi = g1 * (u0_func(x0, y0) - U_ofs)
                if f_func is not None:
                    phi += g2 * f_func(x0, y0)
                G[i, j, :, :] = phi

        # Trapez in beiden Richtungen
        U1 = np.trapz(np.trapz(G, ys, axis=1), xs, axis=0)
        return U_ofs + U1

    def u_(self, x, y, t, u0_func, f_func=None):

        green = self.green
        nx, ny = 100, 100
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
        
        dx, dy = xs[1]-xs[0], ys[1]-ys[0]
        U_ofs = self.apply_bc.c / self.apply_bc.a
        # ---- Anfangsbedingungsteil ----
        U1 = 0.0
        for x0 in xs[1:]:
            for y0 in ys[1:]:
                #print(f"x0: {x0}, y0 {y0}")
                g1, g2 = green(x, y, x0, y0, t)
                U1 += (g1* (u0_func(x0,y0) - U_ofs) + g2*f_func(x0, y0))

        return U_ofs + U1*dx*dy

        # ---- Quellen-Term (zeitunabhängig f(x,y)) ----
        if f_func is None:
            return U_ofs + U1*dx*dy

        U2 = 0.0
        nt = 200  # Anzahl Zeitstufen für zeitliches Integral
        ts = np.linspace(0, t, nt)
        dt = ts[1] - ts[0]
        for s in ts:
            for x0 in xs:
                for y0 in ys:
                    g1, _ = green(x, y, x0, y0, t - s) * f_func(x0, y0)
                    U2 += g1
            

        # return U_ofs + U1*dx*dy
        # return U_ofs + U2*dx*dy*dt
        return U_ofs + (U1 + U2*dt)*dx*dy

    def u__(self, x, y, t, u0_func, f_func=None, nx=101, ny=101, method="simpson"):
        green = self.green
        xs = np.linspace(0, 1, nx)
        ys = np.linspace(0, 1, ny)
        dx, dy = xs[1]-xs[0], ys[1]-ys[0]
        U_ofs = self.apply_bc.c / self.apply_bc.a

        if f_func is None:
            f_func = lambda x0, y0: 0.0

        # Gitter erstellen
        X0, Y0 = np.meshgrid(xs, ys, indexing="ij")

        # green gibt Felder (nx, ny) oder (nx, ny, …) zurück:
        g1, g2 = green(x, y, X0, Y0, t)

        # integrand als Array berechnen
        integrand = g1 * (u0_func(X0, Y0) - U_ofs) + g2 * f_func(X0, Y0)

        # Integration mit Trapez (du kannst hier auch Simpson reinsetzen)
        U1 = np.trapz(np.trapz(integrand, ys, axis=1), xs, axis=0)

        return U_ofs + U1

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
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
            u_means.append(u_mean)
            print(f"Frame {tval:.2f}: u mean={u_mean:.6f}, Time needed {time.time() - start:.4f}")

        return frames, u_means