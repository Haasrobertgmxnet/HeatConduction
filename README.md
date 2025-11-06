# HeatConduction - Numerische Studie zur Wärmeleitung

[🇩🇪 Deutsch](#deutsch) | [🇪🇸 Español](#español) | [🇬🇧 English](#english)

---

## Deutsch

### Überblick
Dieses Projekt implementiert verschiedene numerische Methoden zur Lösung der 2D-Wärmeleitungsgleichung. Es bietet sowohl explizite (Euler) als auch implizite (Crank-Nicolson) Lösungsverfahren mit interaktiven Animationen zum Vergleich der verschiedenen Ansätze.

### Features
- **Explizite Euler-Methode**: Einfache zeitliche Diskretisierung
- **Crank-Nicolson-Verfahren**: Implizite Methode mit besserer Stabilität
- **Animierte Visualisierung**: Echtzeitvergleich beider Methoden
- **Parametervergleich**: Analyse verschiedener Diffusionskoeffizienten (α-Werte)
- **2D-Heatmaps**: Farbcodierte Darstellung der Temperaturverteilung

### Projektstruktur

```
HeatConduction/
├── Solver.Python/              # Hauptverzeichnis der Python-Implementierung
│   ├── Solver.Python.py        # Hauptskript für Solver-Vergleiche
│   ├── explicit_solver.py      # Expliziter Euler-Solver
│   ├── crank_nicolson.py       # Crank-Nicolson (implizit)
│   ├── pinn_solver.py          # Physics-Informed Neural Network
│   ├── green_function.py       # Analytischer Green-Funktionen-Solver
│   ├── boundary_conditions.py  # Randbedingungsklassen
│   ├── frame_data.py           # Gitter- und Zeitkonfiguration
│   ├── ibvp_data.py            # Anfangs-Randwertproblem-Definition
│   ├── plot_tools.py           # Visualisierungs-Werkzeuge
│   └── function_set.py         # Hilfsfunktionen (Kernels, etc.)
├── case3_models/               # Trainierte PINN-Modelle
├── case3_charts/               # Generierte Plots und Animationen
└── README.md                   # Diese Datei
```

### Mathematische Grundlagen

Die 2D-Wärmeleitungsgleichung wird gelöst:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + f(x,y,t)
```

**Wobei:**
- `u(x,y,t)` die Temperatur am Punkt (x,y) zur Zeit t ist
- `α` der Diffusionskoeffizient (thermische Diffusivität) ist
- `f(x,y,t)` eine externe Wärmequelle ist

**Randbedingungen (Robin-Typ):**
```
a·u + b·(∂u/∂n) = c
```

### Implementierte Lösungsmethoden

#### 1. Expliziter Euler-Solver
- **Datei:** `explicit_solver.py`
- **Methode:** Vorwärtsdifferenzen in der Zeit, zentrale Differenzen im Raum
- **Stabilität:** CFL-Bedingung erforderlich: λₓ + λᵧ ≤ 0.5
- **Performance:** Optional Numba-JIT-Beschleunigung
- **Vorteile:** Einfach zu implementieren, geringer Speicherbedarf
- **Nachteile:** Strenge Stabilitätsbedingung limitiert Zeitschrittweite

#### 2. Crank-Nicolson-Solver
- **Datei:** `crank_nicolson.py`
- **Methode:** Implizites θ-Schema (θ=0.5), Trapezregel
- **Stabilität:** Unbedingt stabil
- **Umsetzung:** Sparse-Matrix-Operationen, LU-Faktorisierung
- **Vorteile:** Keine Stabilitätsbeschränkung, 2. Ordnung genau in Zeit
- **Nachteile:** Lösung linearer Gleichungssysteme erforderlich

#### 3. Physics-Informed Neural Network (PINN)
- **Datei:** `pinn_solver.py`
- **Methode:** Tiefes neuronales Netz mit physikalischen Residuen
- **Architektur:** 5 versteckte Schichten, 50 Neuronen pro Schicht
- **Training:** Minimierung von PDE-Residuum + Randbedingungen
- **Vorteile:** Meshfrei, interpoliert auf beliebigen Punkten
- **Nachteile:** Vortraining erforderlich, höhere Rechenlast

#### 4. Green-Funktionen-Solver
- **Datei:** `green_function.py`
- **Methode:** Analytische Lösung via Separationsansatz
- **Basis:** Fourier-Eigenfunktionen mit Robin-BC
- **Vorteile:** Semi-analytisch, hohe Genauigkeit
- **Nachteile:** Auf separierbare Geometrien beschränkt

### Installation

```bash
# Repository klonen
git clone https://github.com/Haasrobertgmxnet/HeatConduction.git
cd HeatConduction/Solver.Python

# Abhängigkeiten installieren
pip install numpy matplotlib scipy torch numba
```

**Benötigte Pakete:**
- `numpy` - Numerische Berechnungen
- `matplotlib` - Visualisierung und Animationen
- `scipy` - Sparse-Matrix-Operationen
- `torch` - PyTorch für PINN
- `numba` - JIT-Kompilierung (optional)

### Verwendung

#### Grundlegendes Beispiel

```python
from Solver.Python import main

# Führt vollständigen Solver-Vergleich aus
main()
```

#### Individueller Solver

```python
from explicit_solver import HeatExplicitSolver
from frame_data import frame1
from ibvp_data import ibvp1

# Solver-Pipeline ausführen
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, 
    frame1, 
    t_steps_per_frame=1000, 
    n_frames=20
)
```

#### Eigenes Problem definieren

```python
from ibvp_data import IBVPData
from frame_data import FrameData

# Anfangsbedingung
def initial_temp(x, y):
    return 25.0 + 100.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# Wärmequelle
def heat_source(x, y, t):
    return 500.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# IBVP-Konfiguration
my_problem = IBVPData(
    alpha=0.1,                    # Diffusionskoeffizient
    heat_source=heat_source,
    initial_u=initial_temp,
    a=0.5, b=1.0, c=12.5         # Robin BC: 0.5u + 1.0(du/dn) = 12.5
)

# Gitterkonfiguration
my_frame = FrameData(
    lx=1.0, ly=1.0,              # Gebietsgröße
    lt=60.0,                     # Simulationszeit
    nx=60, ny=60,                # Gitterpunkte
    nt=288000                    # Zeitschritte
)
```

### Visualisierung

Das Projekt bietet mehrere Visualisierungsmöglichkeiten:

```python
from plot_tools import single_plot, anim_slide

# Einzelner Snapshot
single_plot(u_frames[10], lx=1.0, ly=1.0, 
            title="Temperatur bei t=30s", 
            cmap='hot', isolines=True,
            save_path='output.png')

# Interaktive Animation mit Slider
anim_slide(u_frames, lx=1.0, ly=1.0, 
           title="Wärmeausbreitung", 
           cmap='coolwarm', isolines=True)
```

### Residuenanalyse

Das Hauptskript berechnet automatisch:

1. **PDE-Residuen:** R = ∂u/∂t - α∇²u
2. **Randbedingungs-Residuen:** für alle vier Seiten
3. **Mittelwerte der Temperatur:** über Zeit

```python
# Beispielausgabe
Frame 30.00: mean=28.453621, min=25.123456 @ (0, 0), 
             max=45.678901 @ (15, 15), Time needed 0.0234
```

### Performance-Optimierung

**Für expliziten Solver:**
```python
# Numba-Beschleunigung aktivieren
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1000, 
    n_frames=20,
    use_numba=True  # JIT-Kompilierung
)
```

**Thread-Limitierung (am Dateianfang):**
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
```

### Beispielkonfigurationen

Das Repository enthält vordefinierte Testfälle:

- **ibvp1:** Gauss-förmige Wärmequelle, konstante Anfangstemperatur
- **frame1:** 30×30 Gitter, 60s Simulation
- **frame2:** 30×300 Gitter (anisotrope Auflösung)

### Entwicklerhinweise

Für detaillierte Informationen zu einzelnen Modulen siehe:
- [Solver.Python Dokumentation](Solver.Python/solver_readme_de.md)
- [API-Referenz](...)

### Lizenz

Dieses Projekt ist Open Source und unter der MIT-Lizenz verfügbar.

### Beiträge

Issues und Pull Requests sind willkommen! Bitte öffnen Sie ein Issue für größere Änderungen.

---

## Español

### Descripción General
Este proyecto implementa varios métodos numéricos para resolver la ecuación de difusión de calor 2D. Ofrece esquemas de solución tanto explícitos (Euler) como implícitos (Crank-Nicolson) con animaciones interactivas para comparar diferentes enfoques.

### Características
- **Método Euler Explícito**: Discretización temporal simple
- **Esquema Crank-Nicolson**: Método implícito con mejor estabilidad
- **Visualización Animada**: Comparación en tiempo real de ambos métodos
- **Comparación de Parámetros**: Análisis de diferentes coeficientes de difusión (valores α)
- **Mapas de Calor 2D**: Representación codificada por colores de la distribución de temperatura

### Estructura del Proyecto

```
HeatConduction/
├── Solver.Python/              # Directorio principal de implementación Python
│   ├── Solver.Python.py        # Script principal para comparación de solvers
│   ├── explicit_solver.py      # Solver Euler explícito
│   ├── crank_nicolson.py       # Crank-Nicolson (implícito)
│   ├── pinn_solver.py          # Red Neuronal Informada por Física
│   ├── green_function.py       # Solver analítico de funciones de Green
│   ├── boundary_conditions.py  # Clases de condiciones de frontera
│   ├── frame_data.py           # Configuración de malla y tiempo
│   ├── ibvp_data.py            # Definición de problema de valor inicial-frontera
│   ├── plot_tools.py           # Herramientas de visualización
│   └── function_set.py         # Funciones auxiliares (kernels, etc.)
├── case3_models/               # Modelos PINN entrenados
├── case3_charts/               # Gráficos y animaciones generados
└── README.md                   # Este archivo
```

### Fundamentos Matemáticos

Se resuelve la ecuación de difusión de calor 2D:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + f(x,y,t)
```

**Donde:**
- `u(x,y,t)` es la temperatura en el punto (x,y) en el tiempo t
- `α` es el coeficiente de difusión (difusividad térmica)
- `f(x,y,t)` es una fuente de calor externa

**Condiciones de Frontera (tipo Robin):**
```
a·u + b·(∂u/∂n) = c
```

### Métodos de Solución Implementados

#### 1. Solver Euler Explícito
- **Archivo:** `explicit_solver.py`
- **Método:** Diferencias hacia adelante en tiempo, diferencias centrales en espacio
- **Estabilidad:** Requiere condición CFL: λₓ + λᵧ ≤ 0.5
- **Rendimiento:** Aceleración opcional con Numba-JIT
- **Ventajas:** Fácil de implementar, bajo uso de memoria
- **Desventajas:** Condición de estabilidad estricta limita paso de tiempo

#### 2. Solver Crank-Nicolson
- **Archivo:** `crank_nicolson.py`
- **Método:** Esquema θ implícito (θ=0.5), regla trapezoidal
- **Estabilidad:** Incondicionalmente estable
- **Implementación:** Operaciones con matrices dispersas, factorización LU
- **Ventajas:** Sin restricción de estabilidad, 2º orden preciso en tiempo
- **Desventajas:** Requiere resolver sistemas lineales

#### 3. Red Neuronal Informada por Física (PINN)
- **Archivo:** `pinn_solver.py`
- **Método:** Red neuronal profunda con residuos físicos
- **Arquitectura:** 5 capas ocultas, 50 neuronas por capa
- **Entrenamiento:** Minimización de residuo PDE + condiciones de frontera
- **Ventajas:** Sin malla, interpola en puntos arbitrarios
- **Desventajas:** Requiere pre-entrenamiento, mayor carga computacional

#### 4. Solver de Funciones de Green
- **Archivo:** `green_function.py`
- **Método:** Solución analítica mediante separación de variables
- **Base:** Eigenfunciones de Fourier con BC de Robin
- **Ventajas:** Semi-analítico, alta precisión
- **Desventajas:** Limitado a geometrías separables

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/Haasrobertgmxnet/HeatConduction.git
cd HeatConduction/Solver.Python

# Instalar dependencias
pip install numpy matplotlib scipy torch numba
```

**Paquetes Requeridos:**
- `numpy` - Cálculos numéricos
- `matplotlib` - Visualización y animaciones
- `scipy` - Operaciones con matrices dispersas
- `torch` - PyTorch para PINN
- `numba` - Compilación JIT (opcional)

### Uso

#### Ejemplo Básico

```python
from Solver.Python import main

# Ejecuta comparación completa de solvers
main()
```

#### Solver Individual

```python
from explicit_solver import HeatExplicitSolver
from frame_data import frame1
from ibvp_data import ibvp1

# Ejecutar pipeline del solver
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, 
    frame1, 
    t_steps_per_frame=1000, 
    n_frames=20
)
```

#### Definir Problema Propio

```python
from ibvp_data import IBVPData
from frame_data import FrameData

# Condición inicial
def initial_temp(x, y):
    return 25.0 + 100.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# Fuente de calor
def heat_source(x, y, t):
    return 500.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# Configuración IBVP
my_problem = IBVPData(
    alpha=0.1,                    # Coeficiente de difusión
    heat_source=heat_source,
    initial_u=initial_temp,
    a=0.5, b=1.0, c=12.5         # Robin BC: 0.5u + 1.0(du/dn) = 12.5
)

# Configuración de malla
my_frame = FrameData(
    lx=1.0, ly=1.0,              # Tamaño del dominio
    lt=60.0,                     # Tiempo de simulación
    nx=60, ny=60,                # Puntos de malla
    nt=288000                    # Pasos de tiempo
)
```

### Visualización

El proyecto ofrece varias opciones de visualización:

```python
from plot_tools import single_plot, anim_slide

# Instantánea única
single_plot(u_frames[10], lx=1.0, ly=1.0, 
            title="Temperatura en t=30s", 
            cmap='hot', isolines=True,
            save_path='output.png')

# Animación interactiva con slider
anim_slide(u_frames, lx=1.0, ly=1.0, 
           title="Propagación de calor", 
           cmap='coolwarm', isolines=True)
```

### Análisis de Residuos

El script principal calcula automáticamente:

1. **Residuos PDE:** R = ∂u/∂t - α∇²u
2. **Residuos de Condiciones de Frontera:** para los cuatro lados
3. **Promedios de Temperatura:** a lo largo del tiempo

```python
# Ejemplo de salida
Frame 30.00: mean=28.453621, min=25.123456 @ (0, 0), 
             max=45.678901 @ (15, 15), Time needed 0.0234
```

### Optimización de Rendimiento

**Para solver explícito:**
```python
# Activar aceleración Numba
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1000, 
    n_frames=20,
    use_numba=True  # Compilación JIT
)
```

**Limitación de Threads (al inicio del archivo):**
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
```

### Configuraciones de Ejemplo

El repositorio contiene casos de prueba predefinidos:

- **ibvp1:** Fuente de calor gaussiana, temperatura inicial constante
- **frame1:** Malla 30×30, simulación de 60s
- **frame2:** Malla 30×300 (resolución anisotrópica)

### Notas para Desarrolladores

Para información detallada sobre módulos individuales, consulte:
- [Documentación Solver.Python](Solver.Python/solver_readme_es.md)
- [Referencia API](...)

### Licencia

Este proyecto es código abierto y está disponible bajo la Licencia MIT.

### Contribuciones

¡Issues y Pull Requests son bienvenidos! Por favor, abra un issue para cambios mayores.

---

## English

### Overview
This project implements various numerical methods for solving the 2D heat diffusion equation. It provides both explicit (Euler) and implicit (Crank-Nicolson) solution schemes with interactive animations to compare different approaches.

### Features
- **Explicit Euler Method**: Simple time discretization
- **Crank-Nicolson Scheme**: Implicit method with better stability
- **Animated Visualization**: Real-time comparison of both methods
- **Parameter Comparison**: Analysis of different diffusion coefficients (α values)
- **2D Heatmaps**: Color-coded representation of temperature distribution

### Project Structure

```
HeatConduction/
├── Solver.Python/              # Main Python implementation directory
│   ├── Solver.Python.py        # Main script for solver comparisons
│   ├── explicit_solver.py      # Explicit Euler solver
│   ├── crank_nicolson.py       # Crank-Nicolson (implicit)
│   ├── pinn_solver.py          # Physics-Informed Neural Network
│   ├── green_function.py       # Analytical Green's function solver
│   ├── boundary_conditions.py  # Boundary condition classes
│   ├── frame_data.py           # Grid and time configuration
│   ├── ibvp_data.py            # Initial-boundary value problem definition
│   ├── plot_tools.py           # Visualization tools
│   └── function_set.py         # Helper functions (kernels, etc.)
├── case3_models/               # Trained PINN models
├── case3_charts/               # Generated plots and animations
└── README.md                   # This file
```

### Mathematical Foundation

The 2D heat diffusion equation is solved:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + f(x,y,t)
```

**Where:**
- `u(x,y,t)` is the temperature at point (x,y) at time t
- `α` is the diffusion coefficient (thermal diffusivity)
- `f(x,y,t)` is an external heat source

**Boundary Conditions (Robin type):**
```
a·u + b·(∂u/∂n) = c
```

### Implemented Solution Methods

#### 1. Explicit Euler Solver
- **File:** `explicit_solver.py`
- **Method:** Forward differences in time, central differences in space
- **Stability:** Requires CFL condition: λₓ + λᵧ ≤ 0.5
- **Performance:** Optional Numba-JIT acceleration
- **Advantages:** Easy to implement, low memory usage
- **Disadvantages:** Strict stability condition limits time step size

#### 2. Crank-Nicolson Solver
- **File:** `crank_nicolson.py`
- **Method:** Implicit θ-scheme (θ=0.5), trapezoidal rule
- **Stability:** Unconditionally stable
- **Implementation:** Sparse matrix operations, LU factorization
- **Advantages:** No stability restriction, 2nd order accurate in time
- **Disadvantages:** Requires solving linear systems

#### 3. Physics-Informed Neural Network (PINN)
- **File:** `pinn_solver.py`
- **Method:** Deep neural network with physical residuals
- **Architecture:** 5 hidden layers, 50 neurons per layer
- **Training:** Minimization of PDE residual + boundary conditions
- **Advantages:** Mesh-free, interpolates at arbitrary points
- **Disadvantages:** Requires pre-training, higher computational cost

#### 4. Green's Function Solver
- **File:** `green_function.py`
- **Method:** Analytical solution via separation of variables
- **Basis:** Fourier eigenfunctions with Robin BC
- **Advantages:** Semi-analytical, high accuracy
- **Disadvantages:** Limited to separable geometries

### Installation

```bash
# Clone repository
git clone https://github.com/Haasrobertgmxnet/HeatConduction.git
cd HeatConduction/Solver.Python

# Install dependencies
pip install numpy matplotlib scipy torch numba
```

**Required Packages:**
- `numpy` - Numerical computations
- `matplotlib` - Visualization and animations
- `scipy` - Sparse matrix operations
- `torch` - PyTorch for PINN
- `numba` - JIT compilation (optional)

### Usage

#### Basic Example

```python
from Solver.Python import main

# Run complete solver comparison
main()
```

#### Individual Solver

```python
from explicit_solver import HeatExplicitSolver
from frame_data import frame1
from ibvp_data import ibvp1

# Run solver pipeline
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, 
    frame1, 
    t_steps_per_frame=1000, 
    n_frames=20
)
```

#### Define Custom Problem

```python
from ibvp_data import IBVPData
from frame_data import FrameData

# Initial condition
def initial_temp(x, y):
    return 25.0 + 100.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# Heat source
def heat_source(x, y, t):
    return 500.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.01)

# IBVP configuration
my_problem = IBVPData(
    alpha=0.1,                    # Diffusion coefficient
    heat_source=heat_source,
    initial_u=initial_temp,
    a=0.5, b=1.0, c=12.5         # Robin BC: 0.5u + 1.0(du/dn) = 12.5
)

# Grid configuration
my_frame = FrameData(
    lx=1.0, ly=1.0,              # Domain size
    lt=60.0,                     # Simulation time
    nx=60, ny=60,                # Grid points
    nt=288000                    # Time steps
)
```

### Visualization

The project offers several visualization options:

```python
from plot_tools import single_plot, anim_slide

# Single snapshot
single_plot(u_frames[10], lx=1.0, ly=1.0, 
            title="Temperature at t=30s", 
            cmap='hot', isolines=True,
            save_path='output.png')

# Interactive animation with slider
anim_slide(u_frames, lx=1.0, ly=1.0, 
           title="Heat propagation", 
           cmap='coolwarm', isolines=True)
```

### Residual Analysis

The main script automatically computes:

1. **PDE Residuals:** R = ∂u/∂t - α∇²u
2. **Boundary Condition Residuals:** for all four sides
3. **Temperature Averages:** over time

```python
# Example output
Frame 30.00: mean=28.453621, min=25.123456 @ (0, 0), 
             max=45.678901 @ (15, 15), Time needed 0.0234
```

### Performance Optimization

**For explicit solver:**
```python
# Enable Numba acceleration
frames, u_means = HeatExplicitSolver.pipeline(
    ibvp1, frame1, 
    t_steps_per_frame=1000, 
    n_frames=20,
    use_numba=True  # JIT compilation
)
```

**Thread Limitation (at file beginning):**
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
```

### Example Configurations

The repository contains predefined test cases:

- **ibvp1:** Gaussian heat source, constant initial temperature
- **frame1:** 30×30 grid, 60s simulation
- **frame2:** 30×300 grid (anisotropic resolution)

### Developer Notes

For detailed information on individual modules, see:
- [Solver.Python Documentation](Solver.Python/solver_readme_en.md)
- [API Reference](...)

### License

This project is open source and available under the MIT License.

### Contributions

Issues and Pull Requests are welcome! Please open an issue for major changes.