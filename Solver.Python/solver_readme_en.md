# Solver.Python - Detailed Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Modules and Classes](#modules-and-classes)
4. [Usage Examples](#usage-examples)
5. [Mathematical Details](#mathematical-details)
6. [Performance Optimization](#performance-optimization)
7. [Extension](#extension)

## Overview

`Solver.Python` is a modular Python library for numerically solving the 2D heat diffusion equation. It implements four different solution approaches and provides extensive tools for visualization and residual analysis.

### Key Features

- **Four Solver Implementations:** Explicit, implicit, PINN, Green's functions
- **Flexible Boundary Conditions:** Dirichlet, Neumann, Robin
- **Residual Analysis:** PDE and boundary condition residuals
- **Interactive Visualization:** Animations with slider control
- **Performance Optimized:** Optional Numba-JIT acceleration

## Architecture

### Data Flow

```
IBVPData + FrameData
        ↓
   Solver.pipeline()
        ↓
   [u_frames, u_means]
        ↓
  Visualization / Analysis
```

### Module Structure

```
Solver.Python/
│
├── Core Solvers
│   ├── explicit_solver.py       # HeatExplicitSolver
│   ├── crank_nicolson.py        # HeatCrankNicolsonSolver
│   ├── pinn_solver.py           # HeatPINNSolver
│   └── green_function.py        # GreenFunctionSolver
│
├── Problem Definition
│   ├── ibvp_data.py             # IBVPData
│   ├── frame_data.py            # FrameData
│   └── boundary_conditions.py   # HeatBoundaryCondition
│
├── Utilities
│   ├── plot_tools.py            # Visualization functions
│   └── function_set.py          # Kernel functions
│
└── Main
    └── Solver.Python.py         # Main script for comparisons
```

## Modules and Classes

### 1. explicit_solver.py

#### Class: `HeatExplicitSolver`

**Description:** Implements the explicit Euler method for the heat equation.

**Constructor:**
```python
HeatExplicitSolver(alpha, dx, dy, dt, bc, use_numba=False)
```

**Parameters:**
- `alpha` (float): Diffusion coefficient α
- `dx, dy` (float): Grid spacings in x and y directions
- `dt` (float): Time step size
- `bc` (callable): Boundary condition function bc(u, dx, dy) → u
- `use_numba` (bool): Activates JIT compilation for speed

**Important Methods:**

##### `check_stability() → bool`
Checks the CFL stability condition.

```python
stable = solver.check_stability()
if not stable:
    print("Warning: CFL condition violated!")
```

**Stability Criterion:**
```
λₓ + λᵧ ≤ 0.5
where λₓ = α·dt/dx², λᵧ = α·dt/dy²
```

##### `step(u, f=None) → u_new`
Performs a single time step.

**Parameters:**
- `u` (ndarray): Current temperature field (nx, ny)
- `f` (ndarray, optional): Source term

**Returns:**
- `u_new` (ndarray): Updated field

##### `pipeline(ibvp, frame, t_steps_per_frame, n_frames, use_numba=False)`
Static method for complete time simulation.

**Returns:**
- `frames` (list): Sequence of solution fields
- `u_means` (list): Mean temperatures per frame

**Example:**
```python
from explicit_solver import HeatExplicitSolver
from ibvp_data import ibvp1
from frame_data import frame1

frames, means = HeatExplicitSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1000, 
    n_frames=20,
    use_numba=True
)
```

**Numba Acceleration:**

The `step_numba()` function implements stencil update with JIT compilation:

```python
@njit
def step_numba(u, lamx, lamy, dt, f):
    nx, ny = u.shape
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = (
                u[i, j]
                + lamx * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
                + lamy * (u[i, j+1] - 2*u[i, j] + u[i, j-1])
                - dt * f[i, j]
            )
    return u_new
```

**Performance Tips:**
- First call: Compilation overhead
- Subsequent calls: 10-50x faster
- Optimal for large grids (nx, ny > 100)

---

### 2. crank_nicolson.py

#### Class: `HeatCrankNicolsonSolver`

**Description:** Implements the implicit Crank-Nicolson method (θ=0.5).

**Constructor:**
```python
HeatCrankNicolsonSolver(alpha, dx, dy, dt, nx, ny, nt, robin)
```

**Parameters:**
- `alpha` (float): Diffusion coefficient
- `dx, dy` (float): Grid spacings
- `dt` (float): Time step size
- `nx, ny` (int): Number of grid points
- `nt` (int): Total number of time steps
- `robin` (tuple): (a, b, c) for boundary condition

**Important Attributes:**
- `Lh` (sparse matrix): Discretized Laplacian operator
- `A` (sparse matrix): Left-hand side of implicit system
- `B` (sparse matrix): Right-hand side
- `_factor` (callable): Factorized LU decomposition of A

**Methods:**

##### `build_L_h()`
Constructs the discrete Laplacian operator using Kronecker products.

**Mathematical Background:**

For 1D second derivative with Robin BC:
```
D₁ᵤ = (u_{i+1} - 2u_i + u_{i-1}) / h²
```

With ghost-point elimination for Robin BC:
```
a·u₀ + b·(u₁ - u₋₁)/(2h) = c
→ u₋₁ = u₁ - (2h/b)(c - a·u₀)
```

The 2D operator is constructed as:
```
Lₕ = Iᵧ ⊗ Dₓ + Dᵧ ⊗ Iₓ
```

##### `crank_nicolson_matrices(kappa)`
Builds system matrices A and B:

```
A = I - (1-θ)·dt·κ·Lₕ
B = I + θ·dt·κ·Lₕ
```

For Crank-Nicolson (θ=0.5):
```
(I - 0.5·dt·α·Lₕ)·u^{n+1} = (I + 0.5·dt·α·Lₕ)·u^n + dt·(q + f)
```

##### `step(u, f=None) → u_new`
Solves the linear system for one time step.

**Implementation:**
```python
rhs = B.dot(u_vec) + dt * f_vec + dt * q_total * alpha
u_new_vec = _factor(rhs)  # Uses prefactorized LU
```

**Example:**
```python
from crank_nicolson import HeatCrankNicolsonSolver
from boundary_conditions import HeatBoundaryCondition

bc = HeatBoundaryCondition(a=0.5, b=1.0, c=12.5)
solver = HeatCrankNicolsonSolver(
    alpha=0.1, dx=0.033, dy=0.033, dt=0.0002,
    nx=30, ny=30, nt=288000,
    robin=bc.to_tuple_x()
)

u = initial_field
for step in range(100):
    u = solver.step(u, heat_source)
```

**Debugging Functions:**

The `dbg_matrix_checks(solver)` function enables diagnostics:
- Eigenvalues of B and Lₕ
- Sparsity pattern visualization
- Consistency checks: A + B ≈ 2I
- Singularity check

---

### 3. pinn_solver.py

#### Class: `HeatPINNSolver`

**Description:** Uses a pre-trained Physics-Informed Neural Network for solution.

**PINN Architecture:**

```python
class PINN(nn.Module):
    def __init__(self, layers, neurons, activation=nn.Tanh()):
        # Input: (x, y, t) → 3 neurons
        # Hidden: layers × neurons
        # Output: u(x,y,t) → 1 neuron
```

**Standard Configuration:**
- 5 hidden layers
- 50 neurons per layer
- Tanh activation
- Output shift: +25°C (baseline)

**Pipeline Method:**

```python
frames, u_means = HeatPINNSolver.pipeline(
    ibvp, frame, 
    t_steps_per_frame=1,  # not used
    n_frames=20
)
```

**Prerequisites:**
- Trained model in `case3_models/model`
- PyTorch installed
- CUDA optional (automatic device detection)

**Inference Process:**

1. Load model: `model.load_state_dict(torch.load('case3_models/model'))`
2. Evaluation mode: `model.eval()`
3. Create grid points: `meshgrid(x, y)`
4. For each time point: `u = model(x, y, t)`

**Example:**
```python
# Model must be pre-trained!
from pinn_solver import HeatPINNSolver

frames, means = HeatPINNSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1,
    n_frames=20
)
```

**Notes:**
- Training not included in `pipeline()`
- Training helper functions available: `generate_data()`, `set_seed()`
- For training, see separate PINN training scripts

---

### 4. green_function.py

#### Class: `GreenFunctionSolver`

**Description:** Analytical/semi-analytical solution via eigenfunction expansion.

**Constructor:**
```python
GreenFunctionSolver(alpha, bc, Lx=1.0, Ly=1.0, M=20, N=20)
```

**Parameters:**
- `alpha` (float): Diffusion coefficient
- `bc` (object): Boundary condition object with attributes a, b, c
- `Lx, Ly` (float): Domain size
- `M, N` (int): Number of eigenmodes (currently fixed to 19)

**Mathematical Foundation:**

The solution is represented as:
```
u(x,y,t) = U_amb + Σₘ Σₙ cₘₙ(t) φₘ(x) φₙ(y)
```

**Eigenfunctions φₖ(x):**
```
φₖ(x) = sin(kx) + (k/γ)cos(kx)
where γ = a/b
```

These satisfy the Robin boundary condition:
```
a·φₖ + b·φₖ' = 0  at x=0, x=L
```

**Eigenvalues:**
Predefined in array `eig_vals` (19 values from 0.96 to 56.57).

**Important Methods:**

##### `phi(eig_vals, x) → phi_vals`
Computes eigenfunction matrix.

**Returns:**
- `phi_vals` (ndarray): Shape (M, Nx), where each row is φₖ(x)

##### `green(x, y, x0, y0, tau)`
Computes Green's function and integrated kernel.

**Formula:**
```
G(x,y,τ; x₀,y₀) = Σₘ Σₙ φₘ(x)φₘ(x₀)φₙ(y)φₙ(y₀) exp(-α(k²ₘ + k²ₙ)τ)
```

**Returns:**
- `G`: Green's function for initial condition
- `G_int`: Time-integrated kernel for source term

##### `u(x, y, t, u0_func, f_func=None)`
Main solution method.

**Solution Formula:**
```
u(x,y,t) = U_amb + ∫∫ G(x,y,t; x₀,y₀) [u₀(x₀,y₀) - U_amb] dx₀dy₀
           + ∫₀ᵗ ∫∫ G(x,y,t-s; x₀,y₀) f(x₀,y₀,s) dx₀dy₀ ds
```

**Optimizations:**
- Projections are cached (`_proj_cache`, `_C0`, `_Cf_static`)
- Time factors: `exp(-α·L·t)` and `(1 - exp(-α·L·t))/(α·L)`
- Single calculation only for time-independent f

**Example:**
```python
from green_function import GreenFunctionSolver
import numpy as np

def u0(x, y):
    return 25.0 + 100.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

def f_source(x, y):
    return 500.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

solver = GreenFunctionSolver(alpha=0.1, bc=ibvp1)
x = np.linspace(0, 1, 60)
y = np.linspace(0, 1, 60)
u_solution = solver.u(x, y, t=30.0, u0_func=u0, f_func=f_source)
```

---

### 5. boundary_conditions.py

#### Class: `HeatBoundaryCondition`

**Description:** Manages boundary conditions for finite difference solvers.

**Constructor:**
```python
HeatBoundaryCondition(a, b, c)
```

**Boundary Condition Types:**

1. **Dirichlet** (b ≈ 0):
   ```
   u = c/a  at boundary
   ```

2. **Neumann** (a ≈ 0):
   ```
   ∂u/∂n = c/b  at boundary
   ```

3. **Robin** (a, b ≠ 0):
   ```
   a·u + b·∂u/∂n = c  at boundary
   ```

**Method: `apply(u, dx, dy)`**

Applies boundary conditions to all four sides.

**Implementation (Robin):**
```python
# Left boundary (x=0)
u_new[0,:] = (c*dx + b*u[1,:]) / (b + a*dx)

# Right boundary (x=L)
u_new[-1,:] = (c*dx + b*u[-2,:]) / (b + a*dx)

# Bottom/top boundaries analogously with dy
```

**Derivation:**

Finite difference approximation of normal derivative:
```
∂u/∂n ≈ (u_interior - u_boundary) / h
```

Insert Robin condition:
```
a·u_boundary + b·(u_interior - u_boundary)/h = c
```

Solve for u_boundary:
```
u_boundary = (c·h + b·u_interior) / (b + a·h)
```

**Example:**
```python
from boundary_conditions import HeatBoundaryCondition

# Robin BC: 0.5u + 1.0(du/dn) = 12.5
# → U_amb = 12.5/0.5 = 25°C
bc = HeatBoundaryCondition(a=0.5, b=1.0, c=12.5)

u = temperature_field  # (ny, nx)
u_with_bc = bc.apply(u, dx=0.033, dy=0.033)
```

---

### 6. frame_data.py

#### Class: `FrameData`

**Description:** Dataclass for space-time discretization.

**Attributes:**
```python
@dataclass
class FrameData:
    lx: float   # Domain length in x (m)
    ly: float   # Domain length in y (m)
    lt: float   # Total simulation time (s)
    nx: int     # Number of x grid points
    ny: int     # Number of y grid points
    nt: int     # Number of time steps
```

**Computed Properties:**
- `dx()`: Grid spacing Δx = lx / (nx-1)
- `dy()`: Grid spacing Δy = ly / (ny-1)
- `dt()`: Time step Δt = lt / (nt-1)

**Predefined Configurations:**
```python
# Coarse grid
frame1 = FrameData(1.0, 1.0, 60.0, 30, 30, 288000)

# Medium resolution
frame1 = FrameData(1.0, 1.0, 60.0, 60, 60, 288000)

# Anisotropic grid (tall)
frame2 = FrameData(1.0, 1.0, 60.0, 30, 300, 288000)
```

**CFL Considerations:**

For explicit solver:
```python
frame = FrameData(lx, ly, lt, nx, ny, nt)
dt = frame.dt()
dx = frame.dx()
alpha = 0.1

# Stability criterion
lambda_x = alpha * dt / dx**2
lambda_y = alpha * dt / dy**2
print(f"λₓ + λᵧ = {lambda_x + lambda_y:.4f}")
print(f"Stable if ≤ 0.5")
```

---

### 7. ibvp_data.py

#### Class: `IBVPData`

**Description:** Defines the initial-boundary value problem.

**Constructor:**
```python
IBVPData(alpha, heat_source, initial_u, a, b, c)
```

**Parameters:**
- `alpha` (float): Thermal diffusivity (m²/s)
- `heat_source` (callable): f(x, y, t) or f(x, y)
- `initial_u` (callable): u₀(x, y)
- `a, b, c` (float): Robin coefficients

**Method: `u_amb()`**
Computes ambient temperature from BC:
```python
U_amb = c/a  (if a ≠ 0)
```

**Example Configuration:**
```python
from ibvp_data import IBVPData
from function_set import GaussKernel, ConstantFunc

# Gaussian heat source at (0.5, 0.5)
heat_kernel = GaussKernel(x0=0.5, y0=0.5, sigma=0.1, amplitude=500.0)

# Constant initial temperature
initial = ConstantFunc(25.0)

# Define problem
ibvp = IBVPData(
    alpha=0.1,                      # m²/s
    heat_source=heat_kernel.evaluate,
    initial_u=initial.evaluate,
    a=0.5, b=1.0, c=12.5            # Robin BC
)

print(f"Ambient temperature: {ibvp.u_amb()}°C")  # 25.0°C
```

**Physical Meaning:**

The Robin boundary condition models convective heat transfer:
```
a·u + b·(du/dn) = c
→ k·(du/dn) = h·(u_amb - u)
```

With:
- k: thermal conductivity
- h: convective heat transfer coefficient
- u_amb: ambient temperature

Conversion:
```
a = h/k,  b = 1,  c = h·u_amb/k
```

---

### 8. plot_tools.py

#### Functions

##### `single_plot(u_frame, lx, ly, title, cmap='hot', isolines=False, save_path=None)`

Creates a single snapshot plot.

**Parameters:**
- `u_frame` (ndarray): Temperature field (ny, nx)
- `lx, ly` (float): Domain size
- `title` (str): Plot title
- `cmap` (str): Matplotlib colormap
- `isolines` (bool): Draw isothermal contours
- `save_path` (str, optional): Save path

**Example:**
```python
from plot_tools import single_plot

single_plot(
    u_frames[10], 
    lx=1.0, ly=1.0,
    title="Temperature distribution at t=30s",
    cmap='hot',
    isolines=True,
    save_path='snapshot_t30.png'
)
```

##### `anim_slide(u_frames, lx, ly, title, cmap='hot', isolines=False)`

Interactive animation with slider control.

**Features:**
- Slider for frame navigation
- Play/Stop button for automatic playback
- Dynamic isothermal lines
- Min/max values in colorbar

**Parameters:**
- `u_frames` (list/ndarray): Sequence of temperature fields
- Other parameters same as `single_plot`

**Example:**
```python
from plot_tools import anim_slide

anim_slide(
    frames, 
    lx=1.0, ly=1.0,
    title="Heat propagation over time",
    cmap='coolwarm',
    isolines=True
)
```

**Interaction:**
- Drag slider: Manually navigate through frames
- Play button: Start/stop automatic animation
- Pause time: 0.05s per frame (50ms)

---

## Usage Examples

### Example 1: Complete Solver Comparison

```python
from Solver.Python import main

# Runs all solvers and creates comparison plots
main()
```

**Output:**
- PDE residual plot
- Boundary condition residual plot
- Temperature average plot
- Snapshot images (every 10 frames)
- Animations (Crank-Nicolson, differences, relative errors)

### Example 2: Custom Problem

```python
import numpy as np
from ibvp_data import IBVPData
from frame_data import FrameData
from explicit_solver import HeatExplicitSolver
from plot_tools import anim_slide

# 1. Define problem
def gaussian_pulse(x, y):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return 25.0 + 75.0 * np.exp(-r2 / 0.001)

def no_source(x, y, t):
    return np.zeros_like(x)

my_problem = IBVPData(
    alpha=0.01,
    heat_source=no_source,
    initial_u=gaussian_pulse,
    a=1.0, b=0.0, c=25.0  # Dirichlet BC: u = 25°C
)

# 2. Configure grid
my_grid = FrameData(
    lx=1.0, ly=1.0,
    lt=10.0,
    nx=100, ny=100,
    nt=100000
)

# 3. Run solver
frames, means = HeatExplicitSolver.pipeline(
    my_problem, my_grid,
    t_steps_per_frame=5000,
    n_frames=20,
    use_numba=True
)

# 4. Visualize
anim_slide(frames, 1.0, 1.0, "Diffusion of Gaussian pulse", cmap='hot')
```

### Example 3: Convergence Study

```python
import numpy as np
from explicit_solver import HeatExplicitSolver
from ibvp_data import ibvp1

# Test different resolutions
resolutions = [30, 60, 90, 120]
errors = []

for nx in resolutions:
    frame = FrameData(1.0, 1.0, 60.0, nx, nx, 288000)
    frames, _ = HeatExplicitSolver.pipeline(ibvp1, frame, 1000, 20)
    
    # Error against reference solution
    error = np.linalg.norm(frames[-1] - reference_solution)
    errors.append(error)
    print(f"nx={nx}: error={error:.6e}")

# Estimate convergence rate
import matplotlib.pyplot as plt
plt.loglog(resolutions, errors, 'o-')
plt.xlabel('Grid points nx')
plt.ylabel('L2 error')
plt.title('Convergence study')
plt.grid(True)
plt.show()
```

### Example 4: Residual Analysis

```python
from Solver.Python import compute_pde_residual, boundary_residual

# Run solver
frames, _ = solver.pipeline(ibvp, frame, 1000, 20)

# Calculate PDE residuals
residuals, mean_residuals = compute_pde_residual(frames, frame, ibvp.alpha)

print(f"Maximum PDE residual: {mean_residuals.max():.6e}")
print(f"Mean PDE residual: {mean_residuals.mean():.6e}")

# Boundary condition residuals (last frame)
R_l, R_r, R_b, R_t = boundary_residual(
    frames[-1], frame, 
    k=ibvp.b, h=ibvp.a, u_amb=ibvp.u_amb()
)

print(f"Boundary residuals (mean squared):")
print(f"  Left:   {R_l.mean():.6e}")
print(f"  Right:  {R_r.mean():.6e}")
print(f"  Bottom: {R_b.mean():.6e}")
print(f"  Top:    {R_t.mean():.6e}")
```

## Mathematical Details

### Discretization (Explicit)

**Time derivative (forward difference):**
```
∂u/∂t ≈ (u^{n+1} - u^n) / Δt
```

**Laplacian operator (central differences):**
```
∂²u/∂x² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / Δx²
∂²u/∂y² ≈ (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / Δy²
```

**Update scheme:**
```
u^{n+1}_{i,j} = u^n_{i,j} + λₓ(u^n_{i+1,j} - 2u^n_{i,j} + u^n_{i-1,j})
                        + λᵧ(u^n_{i,j+1} - 2u^n_{i,j} + u^n_{i,j-1})
                        + Δt·f_{i,j}
```

where `λₓ = α·Δt/Δx²` and `λᵧ = α·Δt/Δy²`.

**Stability condition (von Neumann analysis):**
```
λₓ + λᵧ ≤ 1/2
```

### Discretization (Crank-Nicolson)

**θ-scheme:**
```
(u^{n+1} - u^n)/Δt = α[(1-θ)∇²u^n + θ∇²u^{n+1}] + f
```

For θ = 0.5 (Crank-Nicolson):
```
(u^{n+1} - u^n)/Δt = (α/2)[∇²u^n + ∇²u^{n+1}] + f
```

**Matrix form:**
```
(I - θ·Δt·α·Lₕ)u^{n+1} = (I + (1-θ)·Δt·α·Lₕ)u^n + Δt·f
         A                           B
```

**Properties:**
- θ = 0: Explicit Euler (conditionally stable)
- θ = 0.5: Crank-Nicolson (unconditionally stable, 2nd order)
- θ = 1: Implicit Euler (unconditionally stable, 1st order)

### Green's Function Method

**Eigenvalue problem:**
```
-φ''(x) = λφ(x)
a·φ(0) + b·φ'(0) = 0
a·φ(L) + b·φ'(L) = 0
```

**Solution ansatz:**
```
φₖ(x) = sin(kₖx) + (kₖ/γ)cos(kₖx)
```

with γ = a/b and kₖ from eigenvalue equation.

**2D solution:**
```
u(x,y,t) = U_amb + ΣₘΣₙ Aₘₙ exp(-α(k²ₘ + k²ₙ)t) φₘ(x)φₙ(y)
```

Coefficients Aₘₙ from projections:
```
Aₘₙ = ∫∫[u₀(x,y) - U_amb]φₘ(x)φₙ(y) dxdy
```

### PINN Loss Function

**Total loss:**
```
L_total = L_PDE + L_BC + L_IC
```

**PDE residual:**
```
L_PDE = (1/N)Σ|∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) - f|²
```

**Boundary conditions:**
```
L_BC = (1/N_BC)Σ|a·u + b·∂u/∂n - c|²
```

**Initial condition:**
```
L_IC = (1/N_IC)Σ|u(x,y,0) - u₀(x,y)|²
```

Derivatives via automatic differentiation (autograd).

## Performance Optimization

### Numba Acceleration

**Activation:**
```python
frames, means = HeatExplicitSolver.pipeline(
    ibvp, frame, 1000, 20,
    use_numba=True
)
```

**Typical Speed Gains:**
- First execution: 1-2s compilation
- Subsequent: 10-50x faster
- Optimal for nx, ny > 50

**Best Practices:**
- Use for long simulations (nt > 10000)
- Not for single time steps
- Compatible with smaller time steps

### Sparse Matrix Optimization

**Crank-Nicolson uses:**
- CSR format for matrix-vector products
- CSC format for LU factorization
- Prefactorization: one-time LU decomposition

```python
# Once during initialization
self._factor = spla.factorized(self.A