import numpy as np
from boundary_conditions import HeatBoundaryCondition
from explicit_solver import HeatExplicitSolver

# Parameters
nx, ny = 30, 30        # grid points
lx, ly = 1.0, 1.0      # domain size
dx, dy = lx/nx, ly/ny
alpha = 0.01           # diffusion coefficient
dt = 0.0005            # time step
nt = 200               # number of time steps
nts = [0, 50, 100, 150, 200] # number of time steps

# Initial condition: hot spot in the middle
u0 = np.zeros((nx, ny))
u0[nx//2, ny//2] = 1.0

def main() -> None:
    """
    Main execution function that compares explicit and implicit solvers for the 2D heat equation
    """
    print("MAIN")
    neumann_bc = HeatBoundaryCondition(0, 1, 0)
    solver = HeatExplicitSolver(alpha, dx, dy, dt, neumann_bc.apply_robin)

    u = solver.n_steps(u0, None, 5)
    print(u)

if __name__ == "__main__":
    main()
