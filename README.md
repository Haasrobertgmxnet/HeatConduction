# 2D Heat Equation Solver

## Language / Sprache / Idioma:
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

### Mathematischer Hintergrund
Die 2D-Wärmeleitungsgleichung wird gelöst:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

Wobei:
- `u(x,y,t)` die Temperatur am Punkt (x,y) zur Zeit t ist
- `α` der Diffusionskoeffizient ist

### Installation
```bash
pip install numpy matplotlib scipy
```

### Verwendung
```python
from heat_equation_solver import *

# Beispiel 1: Statischer Vergleich
example1()

# Beispiel 2: Animierter Vergleich verschiedener α-Werte
main()
```

### Parameter
- `nx, ny = 30, 30`: Gitterpunkte
- `lx, ly = 1.0, 1.0`: Gebietsgröße
- `α = 0.01`: Diffusionskoeffizient
- `dt = 0.0005`: Zeitschritt
- `nt = 200`: Anzahl Zeitschritte

### Anfangsbedingung
Ein heißer Punkt in der Mitte des Gebiets (`u0[nx//2, ny//2] = 1.0`)

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

### Fundamento Matemático
Se resuelve la ecuación de difusión de calor 2D:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

Donde:
- `u(x,y,t)` es la temperatura en el punto (x,y) en el tiempo t
- `α` es el coeficiente de difusión

### Instalación
```bash
pip install numpy matplotlib scipy
```

### Uso
```python
from heat_equation_solver import *

# Ejemplo 1: Comparación estática
example1()

# Ejemplo 2: Comparación animada de diferentes valores α
main()
```

### Parámetros
- `nx, ny = 30, 30`: Puntos de malla
- `lx, ly = 1.0, 1.0`: Tamaño del dominio
- `α = 0.01`: Coeficiente de difusión
- `dt = 0.0005`: Paso de tiempo
- `nt = 200`: Número de pasos de tiempo

### Condición Inicial
Un punto caliente en el centro del dominio (`u0[nx//2, ny//2] = 1.0`)

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

### Mathematical Background
The 2D heat diffusion equation is solved:

```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```

Where:
- `u(x,y,t)` is the temperature at point (x,y) at time t
- `α` is the diffusion coefficient

### Installation
```bash
pip install numpy matplotlib scipy
```

### Usage
```python
from heat_equation_solver import *

# Example 1: Static comparison
example1()

# Example 2: Animated comparison of different α values
main()
```

### Parameters
- `nx, ny = 30, 30`: Grid points
- `lx, ly = 1.0, 1.0`: Domain size
- `α = 0.01`: Diffusion coefficient
- `dt = 0.0005`: Time step
- `nt = 200`: Number of time steps

### Initial Condition
A hot spot in the center of the domain (`u0[nx//2, ny//2] = 1.0`)

### Methods Implemented

#### Explicit Euler Method
- Forward difference in time
- Central differences in space
- Stability limited by CFL condition

#### Crank-Nicolson Method
- Implicit scheme with trapezoidal rule
- Unconditionally stable
- Uses sparse matrix operations for efficiency

### File Structure
```
├── heat_equation_solver.py    # Main implementation
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

### Dependencies
- `numpy`: Numerical computations
- `matplotlib`: Plotting and animations
- `scipy`: Sparse matrix operations

### Contributing
Feel free to submit issues and pull requests to improve the implementation or add new features.

### License
This project is open source and available under the MIT License.