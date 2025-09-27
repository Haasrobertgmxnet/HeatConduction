import numpy as np
import time
import matplotlib.pyplot as plt
from boundary_conditions import HeatBoundaryCondition
from explicit_solver import HeatExplicitSolver
from pinn_solver import HeatPINNSolver
from frame_data import frame1
from ibvp_data import ibvp1
from plot_tools import anim_slide, single_plot

def main() -> None:
    """
    Main execution function that compares explicit and implicit solvers for the 2D heat equation
    """
    print("MAIN")
    
    n_frames = 50
    start = time.time()
    u_frames_explicit, u_means_explicit = HeatExplicitSolver.pipeline(ibvp1, frame1, frame1.nt//n_frames, n_frames)
    print(f"Time: {(time.time()-start):.4}")
    start = time.time()
    u_frames_pinn, u_means_pinn = HeatPINNSolver.pipeline(ibvp1, frame1, frame1.nt//n_frames, n_frames)
    print(f"Time: {(time.time()-start):.4}")

    if len(u_frames_explicit) != len(u_frames_pinn):
        raise ValueError(f"Unterschiedliche Frameanzahl: {len(u_frames_explicit)} vs {len(u_frames_pinn)}")

    diffs = []
    res = []
    # Vergleich frame für frame
    for i, (f1, f2) in enumerate(zip(u_frames_explicit, u_frames_pinn)):
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


    use_pinn = True
    u_frames = u_frames_explicit
    anim_title = 'Solutions of explicit solver'
    if use_pinn:
        u_frames = u_frames_pinn
        anim_title = 'Solutions of PINN'
    
    print(f"Time: {(time.time()-start):.4}")
    lx = frame1.lx
    ly = frame1.ly

    # Erster Plot
    plt.plot(u_means_pinn, 'go--', linewidth=1, markersize=5, label='u_means_pinn')

    # Zweiter Plot
    plt.plot(u_means_explicit, 'b^-.', linewidth=1, markersize=5, label='u_means_explicit')

    # Optional: Achsenbeschriftung, Legende usw.
    plt.xlabel('Index')
    plt.ylabel('Wert %')
    plt.title('Two in one Chart')
    plt.legend()

    plt.show()

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
