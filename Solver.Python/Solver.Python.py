import numpy as np
import time
import matplotlib.pyplot as plt
from boundary_conditions import HeatBoundaryCondition
from explicit_solver import HeatExplicitSolver
from crank_nicolson import HeatCrankNicolsonSolver
from pinn_solver import HeatPINNSolver
from green_function import GreenFunctionSolver
from frame_data import frame1
from frame_data import frame2
from ibvp_data import ibvp1
from plot_tools import anim_slide, single_plot

def compute_pde_residual(u_frames, frame, alpha):
    dx = frame.lx / (frame.nx - 1)
    dy = frame.ly / (frame.ny - 1)
    dt = frame.lt / frame.nt

    residual_means = []
    residuals = []
    for n in range(1, len(u_frames)-1):
        u_prev_ = u_frames[n-1]
        u = u_frames[n]
        u_next = u_frames[n+1]

        # Zeitliche Ableitung (zentral)
        try:
            u_t = (u_next - u_prev_) / (2*dt)
        except:
            u_t = (u_next - u) / dt

        # Räumliche zweite Ableitungen (Laplace-Operator)
        u_xx = (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2
        u_yy = (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dy**2
        laplace_u = u_xx + u_yy

        R = u_t - alpha * laplace_u
        residuals.append(R)
        residual_means.append(np.mean(R))

    return np.array(residuals), residual_means

def boundary_residual(u, frame, k, h, u_amb):
    dx = frame.lx / (frame.nx - 1)
    dy = frame.ly / (frame.ny - 1)

    # Normalen-Ableitungen an den vier Rändern
    du_dx_left  = (u[:,1] - u[:,0]) / dx
    du_dx_right = (u[:,-1] - u[:,-2]) / dx
    du_dy_bottom = (u[1,:] - u[0,:]) / dy
    du_dy_top    = (u[-1,:] - u[-2,:]) / dy

    # Residuen für Robin-Bedingung
    R_left  = -k*du_dx_left  + h*u[:,0]  - h*u_amb
    R_right =  k*du_dx_right + h*u[:,-1] - h*u_amb
    R_bottom = -k*du_dy_bottom + h*u[0,:] - h*u_amb
    R_top    =  k*du_dy_top    + h*u[-1,:] - h*u_amb

    return R_left, R_right, R_bottom, R_top

def main() -> None:
    """
    Main execution function that compares explicit and implicit solvers for the 2D heat equation
    """
    print("MAIN")
    
    def fplot():
        def func(z):
            gamma = 0.5
            return z*z*np.sin(z)-2*z*gamma*np.cos(z) - gamma*gamma*np.sin(z)

        def solve0(func, cutoff=20, coarse_points=10000, refine_points=1000):
            import numpy as np
            from scipy.optimize import root_scalar

            z_vals = np.linspace(0, 100, num=coarse_points)
            f_vals = func(z_vals)
            sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
            roots = []

            for idx in sign_changes:
                z1, z2 = z_vals[idx], z_vals[idx + 1]

                # Verfeinerung um den Vorzeichenwechsel
                z_refined = np.linspace(z1, z2, refine_points)
                f_refined = func(z_refined)
                sc = np.where(np.diff(np.sign(f_refined)))[0]
                if len(sc) > 0:
                    z1, z2 = z_refined[sc[0]], z_refined[sc[0] + 1]

                try:
                    sol = root_scalar(func, bracket=[z1, z2], method='brentq', xtol=1e-12, rtol=1e-12)
                    roots.append(sol.root)
                except ValueError:
                    pass

            return np.array(roots[:cutoff])

        def solve1(func, cutoff = 20):
            from scipy.optimize import root_scalar
            z_vals = np.linspace(0, 100, num=10000)
            f_vals = func(z_vals)
            sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
            roots = []
            for idx in sign_changes:
                z1, z2 = z_vals[idx], z_vals[idx + 1]
                try:
                    root = root_scalar(func, bracket=[z1, z2], method='brentq').root
                    roots.append(root)
                except ValueError:
                    pass

            return np.array(roots[:cutoff])

        def solve2(func, cutoff = 20):
            from scipy.optimize import newton
            roots = []
            for n in range(1, 40):  # genug Kandidaten
                z0 = n * np.pi
                try:
                    r = newton(func, z0)
                    if all(abs(r - rr) > 1e-3 for rr in roots):  # Duplikate vermeiden
                        roots.append(r)
                except RuntimeError:
                    pass

            return np.array(roots[:cutoff])

        upper_limit = 20
        z = np.linspace(0.0,upper_limit, num=200)
        roots0 = solve0(func, 10)
        roots1 = solve1(func, 10)
        roots2 = solve2(func, 10)
        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(z, func(z), 'g--', linewidth=1, markersize=3, label='Function $ F(\mu )$')
        plt.axhline(0, color='black', lw=1)

        roots3=np.array([n*np.pi for n in range(10)])

        # Nullstellen markieren
        plt.scatter(roots1, func(roots1), color='red', zorder=5, s=47, label='Zeros of $ F(\mu )$')
        for r in roots1[:7]:
            plt.text(r, func(r) + 0.1, f"{r:.4f}", 
                     ha='left', va='bottom', fontsize=12, rotation=45, color='blue')
        # plt.scatter(roots1, np.zeros_like(roots1), color='red', zorder=5, s=35, label='Methode 1')
        # plt.scatter(roots0, func(roots0), color='purple', zorder=5, s=59, label='Methode 0')
        # plt.scatter(roots1, func(roots1), color='red', zorder=5, s=47, label='Methode 1')
        # plt.scatter(roots2, func(roots2), color='blue', zorder=5, s=35, label='Methode 2')
        # plt.scatter(roots3, func(roots3), color='grey', zorder=5, s=23, label='Ganz grob')
        print(f"Roots 1: {roots1}")
        print(f"Roots 0: {roots0}")

        

        def scal(roots):
            gamma=0.5
            sc = np.sin(roots)*np.cos(roots)
            s1 = 0.5*(roots**2 - sc)/roots
            s2 = np.sin(roots)**2/gamma
            s3 = 0.5*(roots**2 + sc) * roots/(gamma**2)
            s = s1+s2+s3
            return np.sqrt(1.0/s)

        print(scal(roots0))

        ref = 0.96018887
        vals = []
        for ele0 in [-5e-6, -4e-6, -3e-6, -2e-6, -1e-6, 0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6]:
            ele = ref + ele0
            vals.append(func(ele))
            print(f"func at {ele:.8f}: {func(ele):.8f}, scal: {scal(ele):.8f}")

        print(np.min(np.abs(vals)))

        # plt.title('Function $ F(\mu ) = (\gamma^2  - \mu^2) \mathrm{sin}(\mu ) + 2 \mu \gamma \mathrm{cos }(\mu )$ and its Zeros')
        plt.xlabel('$ \mu $')
        plt.ylabel('$ F(\mu )$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, upper_limit)
        plt.show()
        print(roots3)

    # fplot()

    class case_data:
        def __init__(self, pipeline, plotsymb = 'go--'):
            self.pipeline = pipeline
            self.u_frames = None
            self.u_means = None
            self.R_pde = None
            self.R_pde_means = None
            self.R_left = None
            self.R_right = None
            self.R_bottom = None
            self.R_top = None
            self.R_left_means = None
            self.R_right_means = None
            self.R_bottom_means = None
            self.R_top_means = None
            self.plotsymb = plotsymb

        def get_solution(self, params):
            self.u_frames, self.u_means = self.pipeline(*params)

        def get_pde_residual(self, frame, alpha):
            self.R_pde, self.R_pde_means = compute_pde_residual(self.u_frames, frame, alpha)
            # print(f"self.R_pde_means: {self.R_pde_means}")

        def get_bdry_residuals(self, frame, ibvp):
            self.R_left, self.R_right, self.R_bottom, self.R_top = boundary_residual(self.u_frames[-1], frame, ibvp.b, ibvp.a, ibvp.u_amb())
            self.R_left_means, self.R_right_means, self.R_bottom_means, self.R_top_means = [np.mean(R) for R in [self.R_left, self.R_right, self.R_bottom, self.R_top]]



    n_frames = 20
    start = time.time()
    params = [ibvp1, frame1, frame1.nt//n_frames, n_frames]

    data_dict_all = dict({"Green" : case_data(GreenFunctionSolver.pipeline, 'go--'),
                      "Explicit" : case_data(HeatExplicitSolver.pipeline, 'b^-.'),
                      "Crank-Nicolson" : case_data(HeatCrankNicolsonSolver.pipeline, 'r*:'),
                      "PINN" : case_data(HeatPINNSolver.pipeline, 'ro--')
                      })

    data_dict_test = dict({"Crank-Nicolson" : case_data(HeatCrankNicolsonSolver.pipeline, 'r*:')
                      })

    data_dict = data_dict_all

    # Erstelle Figure+Axes (korrekte Verwendung)
    fig_pde, ax_pde = plt.subplots(figsize=(8,4))
    fig_bdry_xl, ax_bdry_xl = plt.subplots(figsize=(8,4))

    # Loop über deine Solver-Objekte
    for k, v in data_dict.items():
        print(k)
        # Methoden aufrufen (sollen Attribute wie R_pde_means / R_left_means setzen)
        v.get_solution(params)
        v.get_pde_residual(frame1, ibvp1.alpha)
        v.get_bdry_residuals(frame1, ibvp1)

        # Versuche zuerst die vorgefertigten "means" zu verwenden,
        # falls nicht vorhanden, berechne geeignete Normen aus den Rohdaten.
        R_pde_means = getattr(v, 'R_pde_means', None)
        if R_pde_means is None and hasattr(v, 'R_pde'):
            # z.B. L2-norm pro Frame
            R_pde_means = np.array([np.sqrt(np.mean(r**2)) for r in v.R_pde])

        R_left_means = getattr(v, 'R_left_means', None)
        if R_left_means is None and hasattr(v, 'R_left'):
            # z.B. max-abs über Randpunkte pro Frame
            R_left_means = np.array([np.abs(r) for r in v.R_left])

        # Fallbacks, falls gar nichts gefunden wurde
        if R_pde_means is None:
            raise AttributeError(f"{k}: Keine PDE-Residuen gefunden (R_pde_means oder R_pde).")
        if R_left_means is None:
            raise AttributeError(f"{k}: Keine Rand-Residuen gefunden (R_left_means oder R_left).")

        label = f"{k} residuals"
        # plotsymb kann z.B. '-o', 's--' sein; stelle sicher, dass es existiert
        plot_sym = getattr(v, 'plotsymb', '-o')

        ax_pde.plot(R_pde_means, plot_sym, linewidth=1, markersize=5, label=label)
        ax_bdry_xl.plot([v.R_left_means, v.R_right_means, v.R_bottom_means, v.R_top_means], v.plotsymb, linewidth=1, markersize=5, label=label)
        # ax_bdry_xl.plot([[0,v.R_left_means], [1,v.R_right_means], [2,v.R_bottom_means], [3,v.R_top_means]], v.plotsymb, linewidth=1, markersize=5, label=label)
        # ax_bdry_xl.plot(R_left_means, plot_sym, linewidth=1, markersize=5, label=label)

    # Achsenbeschriftung, Legenden und Grid
    ax_pde.set_xlabel('Timeframe')
    ax_pde.set_ylabel('PDE residual (L2 per frame)')
    ax_pde.set_title('PDE residuals')
    ax_pde.legend()
    ax_pde.grid(True, alpha=0.3)

    ax_bdry_xl.set_xlabel('Timeframe')
    ax_bdry_xl.set_ylabel('Boundary residual (max abs per frame)')
    ax_bdry_xl.set_title('Boundary residuals (left boundary)')
    ax_bdry_xl.legend()
    ax_bdry_xl.grid(True, alpha=0.3)

    # Anzeigen aller Figures
    plt.show()


    plt_pde = plt.figure()
    plt_bdry = plt.figure()
    
    for [k,v] in data_dict.items():
        print(k)
        v.get_solution(params)
        v.get_pde_residual(frame1, ibvp1.alpha)
        v.get_bdry_residuals(frame1, ibvp1)
        s = k+' function'
        plt_pde.plot(v.R_pde_means, v.plotsymb, linewidth=1, markersize=5, label=s)
        plt_bdry.plot([[0,v.R_left_means], [1,v.R_right_means], [2,v.R_bottom_means], [3,v.R_top_means]], v.plotsymb, linewidth=1, markersize=5, label=s)

    # Optional: Achsenbeschriftung, Legende usw.
    plt_pde.xlabel('Timeframe')
    plt_pde.title('PDE residuals')
    plt_pde.legend()

    plt_bdry.xlabel('Timeframe')
    plt_bdry.title('Boundary residuals')
    plt_bdry.legend()

    plt_pde.show()
    plt_bdry.show()

    # Erster Plot
    # plt.plot(data_dict['Green'].R_pde_means, 'go--', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ Green function')

    # Zweiter Plot
    # plt.plot(data_dict['Explicit'].R_pde_means, 'b^-.', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ explicit')

    # Dritter Plot
    # plt.plot(data_dict['Crank-Nicolson'].R_pde_means, 'r*:', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ Green function')

    # Vierter Plot
    # plt.plot(data_dict['PINN'].R_pde_means, 'ro--', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ Crank-Nicolson')

    

    


    # params2 = [ibvp1, frame2, frame2.nt//n_frames, n_frames]
    # u_frames_crank_nicolson, u_means_crank_nicolson = HeatCrankNicolsonSolver.pipeline(ibvp1, frame1, frame1.nt//n_frames, n_frames)
    u_frames_crank_nicolson, u_means_crank_nicolson = HeatCrankNicolsonSolver.pipeline(*params)
    u_frames_green, u_means_green = GreenFunctionSolver.pipeline(*params)
    u_frames_explicit, u_means_explicit = HeatExplicitSolver.pipeline(ibvp1, frame1, frame1.nt//n_frames, n_frames)
    u_frames_pinn, u_means_pinn = HeatPINNSolver.pipeline(ibvp1, frame1, frame1.nt//n_frames, n_frames)

    print(f"Time: {(time.time()-start):.4}")

    if len(u_frames_explicit) != len(u_frames_pinn):
        raise ValueError(f"Unterschiedliche Frameanzahl: {len(u_frames_explicit)} vs {len(u_frames_pinn)}")

    diffs = []
    res = []
    # u_frames = u_frames_explicit # u_frames_crank_nicolson
    u_frames = u_frames_crank_nicolson
    u_means = u_means_crank_nicolson
    # Vergleich frame für frame
    for i, (f1, f2) in enumerate(zip(u_frames_explicit, u_frames)):
        if f1.shape != f2.shape:
            print(f"Frame {i}: unterschiedliche Shape {f1.shape} vs {f2.shape}")
            continue
        diff = f1 - f2
        diffs.append(diff)
        max_diff = np.abs(diff).max()
        mse = np.mean(diff**2)
        re = 100.0*2.0*diff/(f1+f2)
        max_re = np.abs(re).max()
        res.append(re)
        print(f"Frame {i}: MaxDiff={max_diff:.3e}, MSE={mse:.3e}, MaxRE={max_re:5}")


    use_pinn = False
    
    anim_title = 'Solutions of explicit solver'
    if use_pinn:
        u_frames = u_frames_pinn
        anim_title = 'Solutions of PINN'
    
    print(f"Time: {(time.time()-start):.4}")
    lx = frame1.lx
    ly = frame1.ly

    
    # Erster Plot
    plt.plot(u_means_pinn, 'go--', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ PINN')

    # Zweiter Plot
    plt.plot(u_means_explicit, 'b^-.', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ explicit')

    # Dritter Plot
    plt.plot(u_means_green, 'r*:', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ Green function')

    # Vierter Plot
    plt.plot(u_means_crank_nicolson, 'ro--', linewidth=1, markersize=5, label='$T_{\mathrm{avg}}$ Crank-Nicolson')

    # Optional: Achsenbeschriftung, Legende usw.
    plt.xlabel('Timeframe')
    plt.ylabel('$T_{\mathrm{avg}}$ ($^\circ$C)')
    plt.title('Spatial Average Temperatures ($T_{\mathrm{avg}}$)')
    plt.legend()

    plt.show()

    s0 = 'case3_charts/Frame'
    t0 = 'Temperature ($^\circ$C) at time t = '

    for j in range(0,n_frames + 1,10):
        s = s0+ str(j) + '.png'
        t = t0 + str(j*frame1.lt/n_frames) + 's'
        print(s)
        single_plot(u_frames[j], lx, ly, t, cmap ='hot', isolines = True, save_path = s)

    single_plot(u_frames[1], lx, ly, "Temperature at 2nd timestep", cmap ='coolwarm', isolines = True)
    single_plot(u_frames[-2], lx, ly, 'penultimate frame', cmap ='coolwarm', isolines = True)
    single_plot(u_frames[-1], lx, ly, 'last frame', cmap ='coolwarm', isolines = True)
    single_plot(u_frames[-1], lx, ly, 'last frame', cmap ='hot', isolines = True)

    # Run slider animation

    anim_slide(u_frames, frame1.lx, frame1.ly, "Solution", cmap ='coolwarm', isolines = True)
    anim_slide(u_frames, frame1.lx, frame1.ly, "Solution", cmap ='hot', isolines = True)

    anim_slide(diffs, frame1.lx, frame1.ly, "Max Differences")

    anim_slide(res, frame1.lx, frame1.ly, "Max Relative Error", cmap ='hot', isolines = True)

if __name__ == "__main__":
    main()
