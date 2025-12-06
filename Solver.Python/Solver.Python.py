import os

# import frame_data
# Limit multi-threading to keep computations deterministic and avoid CPU overload
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import time
import numpy as np
import matplotlib.pyplot as plt

from explicit_solver import HeatExplicitSolver
from crank_nicolson import HeatCrankNicolsonSolver
from pinn_solver import HeatPINNSolver
from green_function import GreenFunctionSolver

from frame_data import frame1, frame2
from ibvp_data import ibvp1
from plot_tools import anim_slide, single_plot
from calculate_modes import fplot

from result_data import result_data
from result_frames import result_frames

# -------------------------------------------------------------------------
# PDE Residual Computation
# -------------------------------------------------------------------------
def compute_pde_residual(u_frames, u_t_frames, frame, alpha, f = None):
    """
    Computes PDE residuals R = u_t - α∇²u (finite difference approximation).
    u_frames: list of 2D arrays (temperature fields over time)
    """
    dx = frame.dx()
    dy = frame.dy()
    dt = frame.lt/(len(u_frames)-1)
    residuals = []
    mean_residuals = []

    for n in range(1, len(u_frames)-1):
        if u_t_frames is None:
            u_t = (u_frames[n+1] - u_frames[n-1]) / (2*dt)
            #u_t = (u_frames[n+1] - u_frames[n]) / (dt)
            #u_t = (u_frames[n] - u_frames[n-1]) / (dt)
            u_t = u_t[1:-1,1:-1]
        else:
            u_t = (u_t_frames[n])[1:-1,1:-1]

        # Laplace operator
        u_xx = (u_frames[n][2:, 1:-1] - 2*u_frames[n][1:-1, 1:-1] + u_frames[n][:-2, 1:-1]) / dx**2
        u_yy = (u_frames[n][1:-1, 2:] - 2*u_frames[n][1:-1, 1:-1] + u_frames[n][1:-1, :-2]) / dy**2
        lap_u = u_xx + u_yy

        if f is None:
            f = 0*u_t
        R = np.abs(u_t - alpha * lap_u -f[1:-1,1:-1])
        residuals.append(R)
        mean_residuals.append(np.mean(R))

    return np.array(residuals), np.array(mean_residuals)


# -------------------------------------------------------------------------
# Boundary Condition Residuals (Robin)
# -------------------------------------------------------------------------
def boundary_residual(u, frame, k, h, u_amb):
    """
    Computes residuals of Robin condition: k * du/dn + h * u = h * u_amb
    Returns squared residual arrays for each boundary side.
    """
    dx = frame.dx()
    dy = frame.dy()

    du_dx_left   = -(u[1, : ] - u[0, :]) / dx
    du_dx_right  = (u[-1, : ] - u[-2, :]) / dx
    du_dy_bottom = -(u[:, 1] - u[:, 0]) / dy
    du_dy_top    = (u[:, -1] - u[:, -2]) / dy

    R_left = k * du_dx_left   + h * u[0, :]   - h * u_amb
    R_right = k * du_dx_right  + h * u[-1, :]  - h * u_amb
    R_bottom = k * du_dy_bottom + h * u[:, 0]   - h * u_amb
    R_top = k * du_dy_top    + h * u[:, -1]  - h * u_amb

    return R_left**2, R_right**2, R_bottom**2, R_top**2


# -------------------------------------------------------------------------
# Storage class for solver cases
# -------------------------------------------------------------------------
class CaseData:
    """
    Holds solver pipeline and results (solutions and residuals).
    Used to compare numerical methods under the same IBVP settings.
    """
    def __init__(self, pipeline, linestyle, color, marker, markersize=6):
        self.pipeline = pipeline
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.markersize = markersize

        self.results = None

        # self.frame_data = None
        self.u_frames = None
        # self.u_t_frames = None
        # self.lap_frames = None
        self.u_means = None
        self.pde_res = None
        self.R_pde_means = None
        self.R_left_means = None
        self.R_right_means = None
        self.R_bottom_means = None
        self.R_top_means = None
        self.f = None

    def compute_solution(self, params):
        # self.frame_data, self.h = self.pipeline(*params)
        self.results = self.pipeline(*params)
        self.u_frames = self.results.get_u_frames()
        # self.u_t_frames = self.results.get_u_t_frames()
        # lap_frames = [f.laplacian for f in self.frame_data]
        # if any(l is not None for l in lap_frames):
        #    self.lap_frames = lap_frames
        self.u_means = [f.mean() for f in self.u_frames]

    def compute_pde_residuals(self, frame, alpha):
        #self.pde_res, self.R_pde_means = compute_pde_residual(self.results.get_u_frames(), None, frame, alpha, self.results.f)
        #return
        if self.results.has_laplacian and self.results.has_u_t:
            print("self.results.has_laplacian and self.results.has_u_t")
            self.R_pde_means = []
            self.pde_res = []
            for n in range(1, len(self.u_frames)-1):
                u_t = (self.results.get_u_t_frames()[n])[1:-1,1:-1]
                lap_u = (self.results.get_laplacians()[n])[1:-1:,1:-1]
                f = self.results.f[1:-1:,1:-1]
                R = np.abs(u_t - alpha * lap_u - f)
                self.R_pde_means.append(np.mean(R))
                self.pde_res.append(R)
            return
        if self.results.has_u_t:
            print("self.results.has_u_t")
            self.pde_res, self.R_pde_means = compute_pde_residual(self.results.get_u_frames(), self.results.get_u_t_frames(), frame, alpha, self.results.f)
            # self.pde_res, self.R_pde_means = compute_pde_residual(self.results.get_u_frames(), None, frame, alpha, self.results.f)
            return
        print("NOT self.results.has_laplacian and NOT self.results.has_u_t")
        self.pde_res, self.R_pde_means = compute_pde_residual(self.results.get_u_frames(), None, frame, alpha, self.results.f)

    def compute_boundary_residuals(self, frame, ibvp):
        u_amb = ibvp.u_amb()
        n = len(self.u_frames)-1
        if self.results.has_derivs:
            print("self.results.has_u_x and self.results.has_u_y")
            R_left = ibvp.b * ( -self.results.get_u_x_frames()[n][0,:] ) + ibvp.a * self.u_frames[n][0,:] - ibvp.a * u_amb
            R_right = ibvp.b * ( self.results.get_u_x_frames()[n][-1,:] ) + ibvp.a * self.u_frames[n][-1,:] - ibvp.a * u_amb
            R_bottom = ibvp.b * ( -self.results.get_u_y_frames()[n][:,0] ) + ibvp.a * self.u_frames[n][:,0] - ibvp.a * u_amb
            R_top = ibvp.b * ( self.results.get_u_y_frames()[n][:,-1] ) + ibvp.a * self.u_frames[n][:,-1] - ibvp.a * u_amb
            self.R_left_means = np.mean(R_left**2)
            self.R_right_means = np.mean(R_right**2)
            self.R_bottom_means = np.mean(R_bottom**2)
            self.R_top_means = np.mean(R_top**2)
            return
        else:
            print("NOT self.results.has_u_x and self.results.has_u_y")
            R_left, R_right, R_bottom, R_top = boundary_residual(self.u_frames[-9], frame, ibvp.b, ibvp.a, u_amb)
            self.R_left_means, self.R_right_means, self.R_bottom_means, self.R_top_means = \
                map(np.mean, [R_left, R_right, R_bottom, R_top])


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
def main():
    print("Running solver comparison...\n")
    # fplot(100)

    n_frames = 20
    params = (ibvp1, frame1, frame1.nt // n_frames, n_frames)
    start = time.time()

    data_all = {
        "Green": CaseData(GreenFunctionSolver.pipeline, ":",  "#d95f02", '^', 7),
        "Explicit": CaseData(HeatExplicitSolver.pipeline, "-", "#7570b3", 's'),
        "Crank-Nicolson": CaseData(HeatCrankNicolsonSolver.pipeline, "-.", "#e7298a", 'D'),
        "PINN": CaseData(HeatPINNSolver.pipeline, "--", "#1b9e77", 'o')
    }

    name_ref = "Green"
    data_ref  = data_all[name_ref]
    data_ref.compute_solution(params)
    data_ref.compute_pde_residuals(frame1, ibvp1.alpha)
    data_ref.compute_boundary_residuals(frame1, ibvp1)
    print("Reference (Green) PDE Residual Mean:", data_ref.R_pde_means)
    print(f"Reference ({name_ref}) PDE Residual Mean: {[float(x) for x in data_ref.R_pde_means]}")

    data_test = {
        #"Green": data_all["Green"],
        #"Explicit": data_all["Explicit"],
        #"Crank-Nicolson": data_all["Crank-Nicolson"],
        "PINN": data_all["PINN"]
    }
    
    data=data_test
    # Compute solutions + residuals
    for name, case in data.items():
        print(f"Processing: {name}")
        if name_ref == name:
            case = data_ref
            continue
        case.compute_solution(params)
        case.compute_pde_residuals(frame1, ibvp1.alpha)
        case.compute_boundary_residuals(frame1, ibvp1)

    # -------------------------------------------------------------------------
    # Visualization Section (replaces old scattered plot code)
    # -------------------------------------------------------------------------

    plot_on_screen = True
    lx, ly = frame1.lx, frame1.ly
    for name, case in data.items():
        if not plot_on_screen:
            break
        # pde_res = case.pde_res
        u_frames = case.results.get_u_frames()
        diff_to_ref = [u - u_ref for u, u_ref in zip(u_frames, data_ref.u_frames)]
        title = name+ ": PDE Residual"
        anim_slide(case.pde_res, lx, ly, title= title, cmap='coolwarm', label = "PDE Residual", isolines=True)
        title = name+ ": Diff to Reference"
        anim_slide(diff_to_ref, lx, ly, title= title, cmap='coolwarm', label = "Diff to Reference", isolines=True)
        title = name+ ": Solution u"
        anim_slide(u_frames, lx, ly, title= title, cmap='coolwarm', isolines=True)

    def plot_pde_residuals(data, out_to_file= False):
        """Plot mean PDE residuals over time for all solver cases."""
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, case in data.items():
            print(f"Solver: {name}")
            pde_res = [float(x) for x in case.R_pde_means]
            pde_res_mean = np.mean(pde_res)
            pde_res_min = np.min(pde_res)
            pde_res_max = np.max(pde_res)
            pde_res_final = pde_res[-1]
            pde_res_rmse = np.sqrt(np.mean((pde_res - pde_res_mean)**2))
            print(f"PDE residuals: {pde_res}")
            print(f"Mean: {pde_res_mean}, Min: {pde_res_min}, Max: {pde_res_max}, , End: {pde_res_final}, RMSE: {pde_res_rmse}")
            ax.plot(case.R_pde_means,
                    color=case.color, linestyle=case.linestyle, marker=case.marker, 
                    markersize=case.markersize, linewidth=1.5, label=name)

        ax.set_xlabel("Frame Index")
        ax.set_ylabel("PDE Residual (mean value)")
        ax.legend()
        ax.grid(alpha=0.3)
        if not out_to_file:
            ax.set_title("PDE Residuals Over Time")
            plt.show()
            return
        plt.savefig("all_pde_residuals.png", dpi=300, bbox_inches='tight')  # Bild speichern
        plt.close()


    def plot_boundary_residuals(data, out_to_file= False):
        """Plot Robin boundary residuals at final frame."""
        fig, ax = plt.subplots(figsize=(8, 4))
        boundaries = ["Left", "Right", "Bottom", "Top"]

        for name, case in data.items():
            vals = [case.R_left_means, case.R_right_means, case.R_bottom_means, case.R_top_means]
            print(f"Solver: {name}")
            print(f"Boundary residuals: {[float(x) for x in vals]}")
            ax.plot(boundaries, vals,
                    color=case.color, linestyle=case.linestyle,
                    marker=case.marker, markersize=case.markersize, linewidth=1.5,
                    label=name)

        ax.set_ylabel("Boundary Residual (mean squared)")
        ax.legend()
        ax.grid(alpha=0.3)
        if not out_to_file:
            ax.set_title("Boundary Condition Residuals (last frame)")
            plt.show()
            return
        ax.set_title(None)
        plt.savefig("all_bdry_residuals.png", dpi=300, bbox_inches='tight')  # Bild speichern
        plt.close()


    def plot_average_temperatures(data, out_to_file= False):
        """Plot spatial mean temperature evolution."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(7, 5))

        for name, case in data.items():
            plt.plot(case.u_means, 
                     color=case.color, linestyle=case.linestyle, marker=case.marker,
                     markersize=case.markersize, linewidth=2,
                     label=fr"$T_{{avg}}$ {name}")

        plt.xlabel("Frame Index")
        plt.ylabel(r"$T_{avg}$ (°C)")
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        if not out_to_file:
            plt.title(r"Spatial Average Temperature Evolution")
            plt.show()
            return
        plt.savefig("all_avg_temps.png", dpi=300, bbox_inches='tight')
        plt.close()

    # --- Create plots ---
    plot_on_screen = True
    plot_pde_residuals(data, True)
    plot_pde_residuals(data) if plot_on_screen else None
    plot_boundary_residuals(data, True)
    plot_boundary_residuals(data) if plot_on_screen else None
    plot_average_temperatures(data, True)
    plot_average_temperatures(data) if plot_on_screen else None

    # -------------------------------------------------------------------------
    # Snapshot Plots
    # -------------------------------------------------------------------------
    u_frames = data[name_ref].results.get_u_frames()  # reference method
    # u_array_frames = u_frames # [frame.u for frame in u_frames]   # nur die 2D-Felder

    lx, ly = frame1.lx, frame1.ly

    print(f"n_frames: {n_frames}")
    print(f"n_frames: {n_frames}")
    for name, case in data.items():
        print(f"Generating snapshot plots for: {name}")
        u_frames = case.results.get_u_frames()
        for j in [0,1,2,3,5,10,20]:
            print(f"j: {j}")
            title = f"T = {j * frame1.lt / n_frames:.2f} s"
            fname = f"case5_charts/{name}_frame_{j}.png"
            single_plot(u_frames[j], lx, ly, title, cmap='hot', isolines=True, save_path=fname)

    return
    u_frames = data[name_ref].results.get_u_frames()  # reference method
    for j in range(0, n_frames + 1, 10):
        print(f"j: {j}")
        title = f"T = {j * frame1.lt / n_frames:.2f} s"
        fname = f"case3_charts/frame_{j}.png"
        single_plot(u_frames[j], lx, ly, title, cmap='hot', isolines=True, save_path=fname)


    # -------------------------------------------------------------------------
    # Animations
    # -------------------------------------------------------------------------
    anim_slide(u_frames, lx, ly, "Solution (Crank-Nicolson)", cmap='coolwarm', isolines=True)
    anim_slide(u_frames, lx, ly, "Solution (Crank-Nicolson, hot colormap)", cmap='hot', isolines=True)

    # Difference animation (Explicit vs Crank-Nicolson)
    u_exp = data["Explicit"].u_frames
    diffs = [f_exp - f_cn for f_exp, f_cn in zip(u_exp, u_frames)]
    anim_slide(diffs, lx, ly, "Difference (Explicit - CN)")

    # Relative error animation
    res = [100 * 2 * diff / (f_exp + f_cn) for diff, (f_exp, f_cn)
           in zip(diffs, zip(u_exp, u_frames))]
    anim_slide(res, lx, ly, "Relative Error (%)", cmap='hot', isolines=True)

    return

    # --- PDE Residual Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, case in data.items():
        ax.plot(case.R_pde_means, color=case.color, linestyle=case.linestyle,
                marker=case.marker, markersize=case.markersize, label=name)
    ax.set_title("PDE Residuals (mean L2 per frame)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    print(f"Total runtime: {time.time() - start:.2f} seconds")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
