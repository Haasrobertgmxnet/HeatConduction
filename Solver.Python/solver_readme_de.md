# Solver.Python - Detaillierte Dokumentation

## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Architektur](#architektur)
3. [Module und Klassen](#module-und-klassen)
4. [Verwendungsbeispiele](#verwendungsbeispiele)
5. [Mathematische Details](#mathematische-details)
6. [Performance-Optimierung](#performance-optimierung)
7. [Erweiterung](#erweiterung)

## Überblick

`Solver.Python` ist eine modulare Python-Bibliothek zur numerischen Lösung der 2D-Wärmeleitungsgleichung. Sie implementiert vier verschiedene Lösungsansätze und bietet umfangreiche Werkzeuge zur Visualisierung und Residuenanalyse.

### Hauptmerkmale

- **Vier Solver-Implementierungen:** Explizit, implizit, PINN, Green-Funktionen
- **Flexible Randbedingungen:** Dirichlet, Neumann, Robin
- **Residuenanalyse:** PDE und Randbedingungsresiduen
- **Interaktive Visualisierung:** Animationen mit Slider-Steuerung
- **Performance-optimiert:** Optional mit Numba-JIT

## Architektur

### Datenfluss

```
IBVPData + FrameData
        ↓
   Solver.pipeline()
        ↓
   [u_frames, u_means]
        ↓
  Visualisierung / Analyse
```

### Modulstruktur

```
Solver.Python/
│
├── Core Solver
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
│   ├── plot_tools.py            # Visualisierungsfunktionen
│   └── function_set.py          # Kernel-Funktionen
│
└── Main
    └── Solver.Python.py         # Hauptskript für Vergleiche
```

## Module und Klassen

### 1. explicit_solver.py

#### Klasse: `HeatExplicitSolver`

**Beschreibung:** Implementiert das explizite Euler-Verfahren für die Wärmeleitungsgleichung.

**Constructor:**
```python
HeatExplicitSolver(alpha, dx, dy, dt, bc, use_numba=False)
```

**Parameter:**
- `alpha` (float): Diffusionskoeffizient α
- `dx, dy` (float): Gitterabstände in x- und y-Richtung
- `dt` (float): Zeitschrittweite
- `bc` (callable): Randbedingungsfunktion bc(u, dx, dy) → u
- `use_numba` (bool): Aktiviert JIT-Kompilierung für Geschwindigkeit

**Wichtige Methoden:**

##### `check_stability() → bool`
Überprüft die CFL-Stabilitätsbedingung.

```python
stability_ok = solver.check_stability()
if not stability_ok:
    print("Warnung: CFL-Bedingung verletzt!")
```

**Stabilitätskriterium:**
```
λₓ + λᵧ ≤ 0.5
wobei λₓ = α·dt/dx², λᵧ = α·dt/dy²
```

##### `step(u, f=None) → u_new`
Führt einen einzelnen Zeitschritt aus.

**Parameter:**
- `u` (ndarray): Aktuelles Temperaturfeld (nx, ny)
- `f` (ndarray, optional): Quellterm

**Returns:**
- `u_new` (ndarray): Aktualisiertes Feld

##### `n_steps(u, f=None, nt=1) → u`
Führt mehrere Zeitschritte mit Randbedingungen aus.

##### `pipeline(ibvp, frame, t_steps_per_frame, n_frames, use_numba=False)`
Statische Methode für vollständige Zeitsimulation.

**Returns:**
- `frames` (list): Sequenz von Lösungsfeldern
- `u_means` (list): Mittlere Temperaturen pro Frame

**Beispiel:**
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

**Numba-Beschleunigung:**

Die Funktion `step_numba()` implementiert den Stencil-Update mit JIT-Kompilierung:

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

**Performance-Tipps:**
- Bei ersten Aufruf: Kompilierungsoverhead
- Nachfolgende Aufrufe: 10-50x schneller
- Optimal für große Gitter (nx, ny > 100)

---

### 2. crank_nicolson.py

#### Klasse: `HeatCrankNicolsonSolver`

**Beschreibung:** Implementiert das implizite Crank-Nicolson-Verfahren (θ=0.5).

**Constructor:**
```python
HeatCrankNicolsonSolver(alpha, dx, dy, dt, nx, ny, nt, robin)
```

**Parameter:**
- `alpha` (float): Diffusionskoeffizient
- `dx, dy` (float): Gitterabstände
- `dt` (float): Zeitschrittweite
- `nx, ny` (int): Anzahl Gitterpunkte
- `nt` (int): Gesamtanzahl Zeitschritte
- `robin` (tuple): (a, b, c) für Randbedingung

**Wichtige Attribute:**
- `Lh` (sparse matrix): Diskretisierter Laplace-Operator
- `A` (sparse matrix): Linke Seite des impliziten Systems
- `B` (sparse matrix): Rechte Seite
- `_factor` (callable): Faktorisierte LU-Zerlegung von A

**Methoden:**

##### `build_L_h()`
Konstruiert den diskreten Laplace-Operator mittels Kronecker-Produkten.

**Mathematischer Hintergrund:**

Für 1D zweite Ableitung mit Robin-BC:
```
D₁ᵤ = (u_{i+1} - 2u_i + u_{i-1}) / h²
```

Mit Ghost-Point-Elimination für Robin BC:
```
a·u₀ + b·(u₁ - u₋₁)/(2h) = c
→ u₋₁ = u₁ - (2h/b)(c - a·u₀)
```

Der 2D-Operator wird konstruiert als:
```
Lₕ = Iᵧ ⊗ Dₓ + Dᵧ ⊗ Iₓ
```

##### `crank_nicolson_matrices(kappa)`
Baut die Systemmatrizen A und B:

```
A = I - (1-θ)·dt·κ·Lₕ
B = I + θ·dt·κ·Lₕ
```

Für Crank-Nicolson (θ=0.5):
```
(I - 0.5·dt·α·Lₕ)·u^{n+1} = (I + 0.5·dt·α·Lₕ)·u^n + dt·(q + f)
```

##### `step(u, f=None) → u_new`
Löst das lineare Gleichungssystem für einen Zeitschritt.

**Implementierung:**
```python
rhs = B.dot(u_vec) + dt * f_vec + dt * q_total * alpha
u_new_vec = _factor(rhs)  # Verwendet vorfaktorisierte LU
```

**Beispiel:**
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

**Debugging-Funktionen:**

Die Funktion `dbg_matrix_checks(solver)` ermöglicht Diagnosen:
- Eigenwerte von B und Lₕ
- Sparsity-Pattern-Visualisierung
- Konsistenzprüfungen: A + B ≈ 2I
- Singularitätsprüfung

---

### 3. pinn_solver.py

#### Klasse: `HeatPINNSolver`

**Beschreibung:** Verwendet ein vortrainiertes Physics-Informed Neural Network zur Lösung.

**PINN-Architektur:**

```python
class PINN(nn.Module):
    def __init__(self, layers, neurons, activation=nn.Tanh()):
        # Input: (x, y, t) → 3 Neuronen
        # Hidden: layers × neurons
        # Output: u(x,y,t) → 1 Neuron
```

**Standardkonfiguration:**
- 5 versteckte Schichten
- 50 Neuronen pro Schicht
- Tanh-Aktivierung
- Ausgabe-Shift: +25°C (Basislinie)

**Pipeline-Methode:**

```python
frames, u_means = HeatPINNSolver.pipeline(
    ibvp, frame, 
    t_steps_per_frame=1,  # nicht verwendet
    n_frames=20
)
```

**Voraussetzungen:**
- Trainiertes Modell in `case3_models/model`
- PyTorch installiert
- CUDA optional (automatische Geräteerkennung)

**Inferenz-Prozess:**

1. Modell laden: `model.load_state_dict(torch.load('case3_models/model'))`
2. Evaluationsmodus: `model.eval()`
3. Gitterpunkte erstellen: `meshgrid(x, y)`
4. Für jeden Zeitpunkt: `u = model(x, y, t)`

**Beispiel:**
```python
# Modell muss vorher trainiert sein!
from pinn_solver import HeatPINNSolver

frames, means = HeatPINNSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1,
    n_frames=20
)
```

**Hinweise:**
- Training nicht in `pipeline()` enthalten
- Trainings-Hilfsfunktionen verfügbar: `generate_data()`, `set_seed()`
- Für Training siehe separate PINN-Trainings-Skripte

---

### 4. green_function.py

#### Klasse: `GreenFunctionSolver`

**Beschreibung:** Analytische/semi-analytische Lösung mittels Eigenfunktionsentwicklung.

**Constructor:**
```python
GreenFunctionSolver(alpha, bc, Lx=1.0, Ly=1.0, M=20, N=20)
```

**Parameter:**
- `alpha` (float): Diffusionskoeffizient
- `bc` (object): Randbedingungsobjekt mit Attributen a, b, c
- `Lx, Ly` (float): Gebietsgröße
- `M, N` (int): Anzahl Eigenmoden (derzeit fix auf 19)

**Mathematische Grundlage:**

Die Lösung wird dargestellt als:
```
u(x,y,t) = U_amb + Σₘ Σₙ cₘₙ(t) φₘ(x) φₙ(y)
```

**Eigenfunktionen φₖ(x):**
```
φₖ(x) = sin(kx) + (k/γ)cos(kx)
wobei γ = a/b
```

Diese erfüllen die Robin-Randbedingung:
```
a·φₖ + b·φₖ' = 0  an x=0, x=L
```

**Eigenwerte:**
Vordefiniert im Array `eig_vals` (19 Werte von 0.96 bis 56.57).

**Wichtige Methoden:**

##### `phi(eig_vals, x) → phi_vals`
Berechnet Eigenfunktionsmatrix.

**Returns:**
- `phi_vals` (ndarray): Form (M, Nx), wobei jede Zeile φₖ(x) ist

##### `green(x, y, x0, y0, tau)`
Berechnet Green-Funktion und integriertes Kernel.

**Formel:**
```
G(x,y,τ; x₀,y₀) = Σₘ Σₙ φₘ(x)φₘ(x₀)φₙ(y)φₙ(y₀) exp(-α(k²ₘ + k²ₙ)τ)
```

**Returns:**
- `G`: Green-Funktion für Anfangsbedingung
- `G_int`: Zeitintegriertes Kernel für Quellterm

##### `u(x, y, t, u0_func, f_func=None)`
Hauptlösungsmethode.

**Lösungsformel:**
```
u(x,y,t) = U_amb + ∫∫ G(x,y,t; x₀,y₀) [u₀(x₀,y₀) - U_amb] dx₀dy₀
           + ∫₀ᵗ ∫∫ G(x,y,t-s; x₀,y₀) f(x₀,y₀,s) dx₀dy₀ ds
```

**Optimierungen:**
- Projektionen werden gecacht (`_proj_cache`, `_C0`, `_Cf_static`)
- Zeitfaktoren: `exp(-α·L·t)` und `(1 - exp(-α·L·t))/(α·L)`
- Nur einmalige Berechnung bei zeitunabhängigem f

**Beispiel:**
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

#### Klasse: `HeatBoundaryCondition`

**Beschreibung:** Verwaltet Randbedingungen für finite Differenzen-Solver.

**Constructor:**
```python
HeatBoundaryCondition(a, b, c)
```

**Randbedingungstypen:**

1. **Dirichlet** (b ≈ 0):
   ```
   u = c/a  am Rand
   ```

2. **Neumann** (a ≈ 0):
   ```
   ∂u/∂n = c/b  am Rand
   ```

3. **Robin** (a, b ≠ 0):
   ```
   a·u + b·∂u/∂n = c  am Rand
   ```

**Methode: `apply(u, dx, dy)`**

Wendet Randbedingungen auf alle vier Seiten an.

**Implementierung (Robin):**
```python
# Linker Rand (x=0)
u_new[0,:] = (c*dx + b*u[1,:]) / (b + a*dx)

# Rechter Rand (x=L)
u_new[-1,:] = (c*dx + b*u[-2,:]) / (b + a*dx)

# Unterer/oberer Rand analog mit dy
```

**Herleitung:**

Finite-Differenzen-Approximation der Normalenableitung:
```
∂u/∂n ≈ (u_interior - u_boundary) / h
```

Robin-Bedingung einsetzen:
```
a·u_boundary + b·(u_interior - u_boundary)/h = c
```

Nach u_boundary auflösen:
```
u_boundary = (c·h + b·u_interior) / (b + a·h)
```

**Beispiel:**
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

#### Klasse: `FrameData`

**Beschreibung:** Dataclass für Raum-Zeit-Diskretisierung.

**Attribute:**
```python
@dataclass
class FrameData:
    lx: float   # Gebietslänge in x (m)
    ly: float   # Gebietslänge in y (m)
    lt: float   # Gesamtsimulationszeit (s)
    nx: int     # Anzahl x-Gitterpunkte
    ny: int     # Anzahl y-Gitterpunkte
    nt: int     # Anzahl Zeitschritte
```

**Berechnete Eigenschaften:**
- `dx()`: Gitterabstand Δx = lx / (nx-1)
- `dy()`: Gitterabstand Δy = ly / (ny-1)
- `dt()`: Zeitschritt Δt = lt / (nt-1)

**Vordefinierte Konfigurationen:**
```python
# Grobes Gitter
frame1 = FrameData(1.0, 1.0, 60.0, 30, 30, 288000)

# Mittlere Auflösung
frame1 = FrameData(1.0, 1.0, 60.0, 60, 60, 288000)

# Anisotropes Gitter (hoch)
frame2 = FrameData(1.0, 1.0, 60.0, 30, 300, 288000)
```

**CFL-Überlegungen:**

Für expliziten Solver:
```python
frame = FrameData(lx, ly, lt, nx, ny, nt)
dt = frame.dt()
dx = frame.dx()
alpha = 0.1

# Stabilitätskriterium
lambda_x = alpha * dt / dx**2
lambda_y = alpha * dt / dy**2
print(f"λₓ + λᵧ = {lambda_x + lambda_y:.4f}")
print(f"Stabil wenn ≤ 0.5")
```

---

### 7. ibvp_data.py

#### Klasse: `IBVPData`

**Beschreibung:** Definiert das Anfangs-Randwert-Problem.

**Constructor:**
```python
IBVPData(alpha, heat_source, initial_u, a, b, c)
```

**Parameter:**
- `alpha` (float): Thermische Diffusivität (m²/s)
- `heat_source` (callable): f(x, y, t) oder f(x, y)
- `initial_u` (callable): u₀(x, y)
- `a, b, c` (float): Robin-Koeffizienten

**Methode: `u_amb()`**
Berechnet Umgebungstemperatur aus BC:
```python
U_amb = c/a  (falls a ≠ 0)
```

**Beispielkonfiguration:**
```python
from ibvp_data import IBVPData
from function_set import GaussKernel, ConstantFunc

# Gauss-förmige Wärmequelle bei (0.5, 0.5)
heat_kernel = GaussKernel(x0=0.5, y0=0.5, sigma=0.1, amplitude=500.0)

# Konstante Anfangstemperatur
initial = ConstantFunc(25.0)

# Problem definieren
ibvp = IBVPData(
    alpha=0.1,                      # m²/s
    heat_source=heat_kernel.evaluate,
    initial_u=initial.evaluate,
    a=0.5, b=1.0, c=12.5            # Robin BC
)

print(f"Umgebungstemperatur: {ibvp.u_amb()}°C")  # 25.0°C
```

**Physikalische Bedeutung:**

Die Robin-Randbedingung modelliert konvektive Wärmeübertragung:
```
a·u + b·(du/dn) = c
→ k·(du/dn) = h·(u_amb - u)
```

Mit:
- k: Wärmeleitfähigkeit
- h: konvektiver Wärmeübergangskoeffizient
- u_amb: Umgebungstemperatur

Umrechnung:
```
a = h/k,  b = 1,  c = h·u_amb/k
```

---

### 8. plot_tools.py

#### Funktionen

##### `single_plot(u_frame, lx, ly, title, cmap='hot', isolines=False, save_path=None)`

Erstellt einzelnen Snapshot-Plot.

**Parameter:**
- `u_frame` (ndarray): Temperaturfeld (ny, nx)
- `lx, ly` (float): Gebietsgröße
- `title` (str): Plot-Titel
- `cmap` (str): Matplotlib-Colormap
- `isolines` (bool): Isotherme Konturen zeichnen
- `save_path` (str, optional): Speicherpfad

**Beispiel:**
```python
from plot_tools import single_plot

single_plot(
    u_frames[10], 
    lx=1.0, ly=1.0,
    title="Temperaturverteilung bei t=30s",
    cmap='hot',
    isolines=True,
    save_path='snapshot_t30.png'
)
```

##### `anim_slide(u_frames, lx, ly, title, cmap='hot', isolines=False)`

Interaktive Animation mit Slider-Steuerung.

**Features:**
- Slider zur Frame-Navigation
- Play/Stop-Button für automatische Wiedergabe
- Dynamische Isothermenlinien
- Min/Max-Werte im Colorbar

**Parameter:**
- `u_frames` (list/ndarray): Sequenz von Temperaturfeldern
- Weitere Parameter wie `single_plot`

**Beispiel:**
```python
from plot_tools import anim_slide

anim_slide(
    frames, 
    lx=1.0, ly=1.0,
    title="Wärmeausbreitung im Zeitverlauf",
    cmap='coolwarm',
    isolines=True
)
```

**Interaktion:**
- Slider ziehen: Manuell durch Frames navigieren
- Play-Button: Automatische Animation starten/stoppen
- Pause-Zeit: 0.05s pro Frame (50ms)

---

## Verwendungsbeispiele

### Beispiel 1: Vollständiger Solver-Vergleich

```python
from Solver.Python import main

# Führt alle Solver aus und erstellt Vergleichsplots
main()
```

**Ausgabe:**
- PDE-Residuen-Plot
- Randbedingungsresiduen-Plot
- Temperaturmittelwerte-Plot
- Snapshot-Bilder (alle 10 Frames)
- Animationen (Crank-Nicolson, Differenzen, relative Fehler)

### Beispiel 2: Eigenes Problem

```python
import numpy as np
from ibvp_data import IBVPData
from frame_data import FrameData
from explicit_solver import HeatExplicitSolver
from plot_tools import anim_slide

# 1. Problem definieren
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

# 2. Gitter konfigurieren
my_grid = FrameData(
    lx=1.0, ly=1.0,
    lt=10.0,
    nx=100, ny=100,
    nt=100000
)

# 3. Solver ausführen
frames, means = HeatExplicitSolver.pipeline(
    my_problem, my_grid,
    t_steps_per_frame=5000,
    n_frames=20,
    use_numba=True
)

# 4. Visualisieren
anim_slide(frames, 1.0, 1.0, "Diffusion eines Gauss-Pulses", cmap='hot')
```

### Beispiel 3: Konvergenzstudien

```python
import numpy as np
from explicit_solver import HeatExplicitSolver
from ibvp_data import ibvp1

# Verschiedene Auflösungen testen
resolutions = [30, 60, 90, 120]
errors = []

for nx in resolutions:
    frame = FrameData(1.0, 1.0, 60.0, nx, nx, 288000)
    frames, _ = HeatExplicitSolver.pipeline(ibvp1, frame, 1000, 20)
    
    # Fehler gegen Referenzlösung
    error = np.linalg.norm(frames[-1] - reference_solution)
    errors.append(error)
    print(f"nx={nx}: error={error:.6e}")

# Konvergenzrate schätzen
import matplotlib.pyplot as plt
plt.loglog(resolutions, errors, 'o-')
plt.xlabel('Grid points nx')
plt.ylabel('L2 error')
plt.title('Convergence study')
plt.grid(True)
plt.show()
```

### Beispiel 4: Residuenanalyse

```python
from Solver.Python import compute_pde_residual, boundary_residual

# Solver ausführen
frames, _ = solver.pipeline(ibvp, frame, 1000, 20)

# PDE-Residuen berechnen
residuals, mean_residuals = compute_pde_residual(frames, frame, ibvp.alpha)

print(f"Maximales PDE-Residuum: {mean_residuals.max():.6e}")
print(f"Mittleres PDE-Residuum: {mean_residuals.mean():.6e}")

# Randbedingungsresiduen (letzter Frame)
R_l, R_r, R_b, R_t = boundary_residual(
    frames[-1], frame, 
    k=ibvp.b, h=ibvp.a, u_amb=ibvp.u_amb()
)

print(f"Randresiduen (mittlere quadratische):")
print(f"  Links:  {R_l.mean():.6e}")
print(f"  Rechts: {R_r.mean():.6e}")
print(f"  Unten:  {R_b.mean():.6e}")
print(f"  Oben:   {R_t.mean():.6e}")
```

## Mathematische Details

### Diskretisierung (Explizit)

**Zeitableitung (Vorwärtsdifferenz):**
```
∂u/∂t ≈ (u^{n+1} - u^n) / Δt
```

**Laplace-Operator (zentrale Differenzen):**
```
∂²u/∂x² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / Δx²
∂²u/∂y² ≈ (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / Δy²
```

**Update-Schema:**
```
u^{n+1}_{i,j} = u^n_{i,j} + λₓ(u^n_{i+1,j} - 2u^n_{i,j} + u^n_{i-1,j})
                        + λᵧ(u^n_{i,j+1} - 2u^n_{i,j} + u^n_{i,j-1})
                        + Δt·f_{i,j}
```

wobei `λₓ = α·Δt/Δx²` und `λᵧ = α·Δt/Δy²`.

**Stabilitätsbedingung (von-Neumann-Analyse):**
```
λₓ + λᵧ ≤ 1/2
```

### Diskretisierung (Crank-Nicolson)

**θ-Schema:**
```
(u^{n+1} - u^n)/Δt = α[(1-θ)∇²u^n + θ∇²u^{n+1}] + f
```

Für θ = 0.5 (Crank-Nicolson):
```
(u^{n+1} - u^n)/Δt = (α/2)[∇²u^n + ∇²u^{n+1}] + f
```

**Matrixform:**
```
(I - θ·Δt·α·Lₕ)u^{n+1} = (I + (1-θ)·Δt·α·Lₕ)u^n + Δt·f
         A                           B
```

**Eigenschaften:**
- θ = 0: Explizites Euler (bedingt stabil)
- θ = 0.5: Crank-Nicolson (unbedingt stabil, 2. Ordnung)
- θ = 1: Implizites Euler (unbedingt stabil, 1. Ordnung)

### Green-Funktionen-Methode

**Eigenwertproblem:**
```
-φ''(x) = λφ(x)
a·φ(0) + b·φ'(0) = 0
a·φ(L) + b·φ'(L) = 0
```

**Lösungsansatz:**
```
φₖ(x) = sin(kₖx) + (kₖ/γ)cos(kₖx)
```

mit γ = a/b und kₖ aus Eigenwertgleichung.

**2D-Lösung:**
```
u(x,y,t) = U_amb + ΣₘΣₙ Aₘₙ exp(-α(k²ₘ + k²ₙ)t) φₘ(x)φₙ(y)
```

Koeffizienten Aₘₙ aus Projektionen:
```
Aₘₙ = ∫∫[u₀(x,y) - U_amb]φₘ(x)φₙ(y) dxdy
```

### PINN-Loss-Funktion

**Gesamt-Loss:**
```
L_total = L_PDE + L_BC + L_IC
```

**PDE-Residuum:**
```
L_PDE = (1/N)Σ|∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) - f|²
```

**Randbedingungen:**
```
L_BC = (1/N_BC)Σ|a·u + b·∂u/∂n - c|²
```

**Anfangsbedingung:**
```
L_IC = (1/N_IC)Σ|u(x,y,0) - u₀(x,y)|²
```

Ableitungen mittels Automatic Differentiation (autograd).

## Performance-Optimierung

### Numba-Beschleunigung

**Aktivierung:**
```python
frames, means = HeatExplicitSolver.pipeline(
    ibvp, frame, 1000, 20,
    use_numba=True
)
```

**Typische Geschwindigkeitsgewinne:**
- Erste Ausführung: 1-2s Kompilierung
- Nachfolgende: 10-50x schneller
- Optimal bei nx, ny > 50

**Best Practices:**
- Nutzen für lange Simulationen (nt > 10000)
- Nicht für einzelne Zeitschritte
- Kombinierbar mit kleineren Zeitschritten

### Sparse-Matrix-Optimierung

**Crank-Nicolson nutzt:**
- CSR-Format für Matrix-Vektor-Produkte
- CSC-Format für LU-Faktorisierung
- Vorfaktorisierung: einmalige LU-Zerlegung

```python
# Einmalig beim Initialisieren
self._factor = spla.factorized(self.A)

# Schnelles Lösen in jedem Zeitschritt
u_new = self._factor(rhs)
```

**Speichereinsparung:**
- Vollmatrix: O(n²) mit n = nx·ny
- Sparse: O(5n) (5-Punkt-Stencil)
- Beispiel 100×100: 10⁴ → 5·10⁴ Elemente (Faktor 0.05)

### Thread-Kontrolle

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
```

**Warum?**
- Deterministische Ergebnisse
- Vermeidung von CPU-Überlastung
- Bessere Performance bei mehreren parallelen Jobs

### Memory-Management

**Große Simulationen:**
```python
# Frames selektiv speichern
frames = []
for n in range(n_frames):
    u = solver.n_steps(u, f, steps_per_frame)
    if n % 10 == 0:  # Nur jedes 10. Frame
        frames.append(u.copy())
```

**Streaming-Output:**
```python
# Direkt auf Disk schreiben
for n in range(n_frames):
    u = solver.n_steps(u, f, steps_per_frame)
    np.save(f'output/frame_{n:04d}.npy', u)
```

## Erweiterung

### Neuen Solver hinzufügen

**Schritt 1: Solver-Klasse erstellen**

```python
# my_solver.py
class MyCustomSolver:
    def __init__(self, alpha, dx, dy, dt, bc):
        self.alpha = alpha
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.apply_bc = bc
    
    def step(self, u, f=None):
        # Implementiere einen Zeitschritt
        u_new = ... # Ihre Methode hier
        return u_new
    
    def n_steps(self, u, f=None, nt=1):
        for _ in range(nt):
            u = self.step(u, f)
            u = self.apply_bc(u, self.dx, self.dy)
        return u
    
    @staticmethod
    def pipeline(ibvp, frame, t_steps_per_frame, n_frames):
        # Vollständige Simulation
        # ... Setup-Code ...
        
        solver = MyCustomSolver(alpha, dx, dy, dt, bc)
        frames = [u0]
        u_means = []
        
        for n in range(n_frames):
            u = solver.n_steps(u, f, t_steps_per_frame)
            frames.append(u.copy())
            u_means.append(u.mean())
            # Logging ...
        
        return frames, u_means
```

**Schritt 2: Integration in Vergleich**

```python
# In Solver.Python.py
from my_solver import MyCustomSolver

data = {
    "My Method": CaseData(MyCustomSolver.pipeline, "-", "#ff7f00", 'v'),
    # ... andere Solver ...
}
```

### Neue Randbedingungen

**Gemischte Randbedingungen:**

```python
class MixedBoundaryCondition:
    def __init__(self, left_bc, right_bc, bottom_bc, top_bc):
        self.left = left_bc    # (a, b, c)
        self.right = right_bc
        self.bottom = bottom_bc
        self.top = top_bc
    
    def apply(self, u, dx, dy):
        u_new = u.copy()
        
        # Linker Rand
        a, b, c = self.left
        if abs(b) < 1e-14:
            u_new[0,:] = c/a
        else:
            u_new[0,:] = (c*dx + b*u[1,:]) / (b + a*dx)
        
        # Analog für andere Seiten
        # ...
        
        return u_new
```

**Zeitabhängige Randbedingungen:**

```python
class TimeDependentBC:
    def __init__(self, bc_func):
        self.bc_func = bc_func  # bc_func(t) → (a, b, c)
    
    def apply(self, u, dx, dy, t):
        a, b, c = self.bc_func(t)
        # Wie HeatBoundaryCondition.apply
        # ...
```

### Neue Quellterme

**Zeitabhängige Quelle:**

```python
class PulsedSource:
    def __init__(self, x0, y0, sigma, freq, amplitude):
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.freq = freq  # Pulsfrequenz (Hz)
        self.amp = amplitude
    
    def evaluate(self, x, y, t):
        r2 = (x - self.x0)**2 + (y - self.y0)**2
        spatial = np.exp(-r2 / self.sigma**2)
        temporal = np.sin(2*np.pi*self.freq*t)
        return self.amp * spatial * temporal
```

**Mehrere Quellen:**

```python
class MultiSource:
    def __init__(self, sources):
        self.sources = sources  # Liste von Source-Objekten
    
    def evaluate(self, x, y, t=0):
        total = np.zeros_like(x)
        for source in self.sources:
            total += source.evaluate(x, y, t)
        return total

# Verwendung
s1 = GaussKernel(0.3, 0.3, 0.05, 200)
s2 = GaussKernel(0.7, 0.7, 0.05, 300)
multi = MultiSource([s1, s2])

ibvp = IBVPData(
    alpha=0.1,
    heat_source=multi.evaluate,
    # ...
)
```

### Adaptive Zeitschrittweite

```python
class AdaptiveExplicitSolver(HeatExplicitSolver):
    def __init__(self, alpha, dx, dy, dt_max, bc, cfl_target=0.4):
        super().__init__(alpha, dx, dy, dt_max, bc)
        self.dt_max = dt_max
        self.cfl_target = cfl_target
    
    def adaptive_step(self, u, f=None):
        # Lokale Zeitableitung schätzen
        u_t = self.step(u, f) - u
        max_change = np.abs(u_t).max()
        
        # Zeitschritt anpassen
        if max_change > 0:
            dt_safe = self.cfl_target * self.dx**2 / (self.alpha * max_change)
            self.dt = min(dt_safe, self.dt_max)
            self.lamx = self.alpha * self.dt / self.dx**2
            self.lamy = self.alpha * self.dt / self.dy**2
        
        return self.step(u, f)
```

### Parallele Ausführung

**Mehrere Solver parallel:**

```python
from multiprocessing import Pool

def run_solver(params):
    solver_class, ibvp, frame = params
    return solver_class.pipeline(ibvp, frame, 1000, 20)

solvers = [
    (HeatExplicitSolver, ibvp1, frame1),
    (HeatCrankNicolsonSolver, ibvp1, frame1),
    (GreenFunctionSolver, ibvp1, frame1),
]

with Pool(3) as pool:
    results = pool.map(run_solver, solvers)
```

### Diagnose-Tools

**Matrix-Sparsity-Analyse:**

```python
from crank_nicolson import dbg_matrix_checks

solver = HeatCrankNicolsonSolver(...)
dbg_matrix_checks(solver)
```

**Ausgabe:**
- Eigenwerte von B und Lₕ
- Sparsity-Pattern (optional mit Plots)
- Konsistenzprüfungen
- Singularitätsprüfung

**Energie-Monitoring:**

```python
def thermal_energy(u, dx, dy):
    """Berechnet Gesamtwärmeenergie."""
    return np.sum(u) * dx * dy

energies = []
for frame in frames:
    E = thermal_energy(frame, dx, dy)
    energies.append(E)

# Energie sollte bei Dirichlet-BC monoton fallen
import matplotlib.pyplot as plt
plt.plot(energies)
plt.xlabel('Frame')
plt.ylabel('Thermal Energy')
plt.show()
```

---

## Häufige Fehler und Lösungen

### Fehler 1: CFL-Verletzung

**Symptom:**
```
Frame 5.00: mean=28.453, min=-1e10, max=1e10
```

**Ursache:** Zeitschritt zu groß für expliziten Solver.

**Lösung:**
```python
# Zeitschritte erhöhen oder Gitter verfeinern
dt_max = 0.5 * dx**2 / alpha
nt = int(lt / dt_max) + 1
frame = FrameData(lx, ly, lt, nx, ny, nt)
```

### Fehler 2: Singular Matrix

**Symptom:**
```
LinAlgError: singular matrix
```

**Ursache:** Falsche Randbedingungen oder degenerierte Geometrie.

**Lösung:**
```python
# Prüfe Randbedingungen
bc = HeatBoundaryCondition(a, b, c)
if abs(a) < 1e-14 and abs(b) < 1e-14:
    print("Fehler: a und b beide null!")
```

### Fehler 3: Langsame Konvergenz

**Symptom:** Simulation läuft stundenlang.

**Lösungen:**
1. Numba aktivieren: `use_numba=True`
2. Größere Zeitschritte (Crank-Nicolson)
3. Gröberes Gitter für Tests
4. Weniger Ausgabe-Frames

### Fehler 4: Speicherfehler

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Lösungen:**
```python
# Frame-Liste reduzieren
frames = []
for n in range(n_frames):
    u = solver.n_steps(u, f, steps)
    if n % 5 == 0:  # Nur jedes 5. Frame speichern
        frames.append(u.copy())

# Oder dtype reduzieren
u = u.astype(np.float32)  # statt float64
```

---

## Referenzen und Weiterführendes

### Literatur

1. **Finite Differenzen:**
   - LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*
   
2. **Crank-Nicolson:**
   - Crank, J., & Nicolson, P. (1947). *A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type*

3. **PINNs:**
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems*

4. **Green-Funktionen:**
   - Duffy, D. G. (2015). *Green's Functions with Applications*

### Weiterführende Themen

- **AMR (Adaptive Mesh Refinement):** Dynamische Gitterverfeinerung
- **Multigrid-Methoden:** Schnellere Lösung impliziter Systeme
- **Operator-Splitting:** Behandlung von Konvektion + Diffusion
- **Higher-Order Schemes:** WENO, DG-Methoden

### Online-Ressourcen

- [NumPy Dokumentation](https://numpy.org/doc/)
- [SciPy Sparse Matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Numba User Guide](https://numba.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## Lizenz und Kontakt

Dieses Projekt ist Open Source unter der MIT-Lizenz.

**Repository:** https://github.com/Haasrobertgmxnet/HeatConduction

**Issues und Beiträge** sind willkommen!