# Solver.Python - Documentación Detallada

## Índice

1. [Descripción General](#descripción-general)
2. [Arquitectura](#arquitectura)
3. [Módulos y Clases](#módulos-y-clases)
4. [Ejemplos de Uso](#ejemplos-de-uso)
5. [Detalles Matemáticos](#detalles-matemáticos)
6. [Optimización de Rendimiento](#optimización-de-rendimiento)
7. [Extensión](#extensión)

## Descripción General

`Solver.Python` es una biblioteca modular de Python para la solución numérica de la ecuación de difusión de calor 2D. Implementa cuatro enfoques de solución diferentes y proporciona herramientas extensas para visualización y análisis de residuos.

### Características Principales

- **Cuatro Implementaciones de Solvers:** Explícito, implícito, PINN, funciones de Green
- **Condiciones de Frontera Flexibles:** Dirichlet, Neumann, Robin
- **Análisis de Residuos:** Residuos PDE y condiciones de frontera
- **Visualización Interactiva:** Animaciones con control deslizante
- **Optimizado para Rendimiento:** Opcional con Numba-JIT

## Arquitectura

### Flujo de Datos

```
IBVPData + FrameData
        ↓
   Solver.pipeline()
        ↓
   [u_frames, u_means]
        ↓
  Visualización / Análisis
```

### Estructura de Módulos

```
Solver.Python/
│
├── Core Solvers
│   ├── explicit_solver.py       # HeatExplicitSolver
│   ├── crank_nicolson.py        # HeatCrankNicolsonSolver
│   ├── pinn_solver.py           # HeatPINNSolver
│   └── green_function.py        # GreenFunctionSolver
│
├── Definición del Problema
│   ├── ibvp_data.py             # IBVPData
│   ├── frame_data.py            # FrameData
│   └── boundary_conditions.py   # HeatBoundaryCondition
│
├── Utilidades
│   ├── plot_tools.py            # Funciones de visualización
│   └── function_set.py          # Funciones kernel
│
└── Principal
    └── Solver.Python.py         # Script principal para comparaciones
```

## Módulos y Clases

### 1. explicit_solver.py

#### Clase: `HeatExplicitSolver`

**Descripción:** Implementa el método Euler explícito para la ecuación de calor.

**Constructor:**
```python
HeatExplicitSolver(alpha, dx, dy, dt, bc, use_numba=False)
```

**Parámetros:**
- `alpha` (float): Coeficiente de difusión α
- `dx, dy` (float): Espaciamientos de malla en direcciones x e y
- `dt` (float): Tamaño de paso de tiempo
- `bc` (callable): Función de condición de frontera bc(u, dx, dy) → u
- `use_numba` (bool): Activa compilación JIT para velocidad

**Métodos Importantes:**

##### `check_stability() → bool`
Verifica la condición de estabilidad CFL.

```python
estable = solver.check_stability()
if not estable:
    print("¡Advertencia: Condición CFL violada!")
```

**Criterio de Estabilidad:**
```
λₓ + λᵧ ≤ 0.5
donde λₓ = α·dt/dx², λᵧ = α·dt/dy²
```

##### `step(u, f=None) → u_new`
Ejecuta un solo paso de tiempo.

**Parámetros:**
- `u` (ndarray): Campo de temperatura actual (nx, ny)
- `f` (ndarray, opcional): Término fuente

**Retorna:**
- `u_new` (ndarray): Campo actualizado

##### `pipeline(ibvp, frame, t_steps_per_frame, n_frames, use_numba=False)`
Método estático para simulación temporal completa.

**Retorna:**
- `frames` (list): Secuencia de campos de solución
- `u_means` (list): Temperaturas medias por frame

**Ejemplo:**
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

---

### 2. crank_nicolson.py

#### Clase: `HeatCrankNicolsonSolver`

**Descripción:** Implementa el método Crank-Nicolson implícito (θ=0.5).

**Constructor:**
```python
HeatCrankNicolsonSolver(alpha, dx, dy, dt, nx, ny, nt, robin)
```

**Parámetros:**
- `alpha` (float): Coeficiente de difusión
- `dx, dy` (float): Espaciamientos de malla
- `dt` (float): Tamaño de paso de tiempo
- `nx, ny` (int): Número de puntos de malla
- `nt` (int): Número total de pasos de tiempo
- `robin` (tuple): (a, b, c) para condición de frontera

**Atributos Importantes:**
- `Lh` (sparse matrix): Operador Laplaciano discretizado
- `A` (sparse matrix): Lado izquierdo del sistema implícito
- `B` (sparse matrix): Lado derecho
- `_factor` (callable): Descomposición LU factorizada de A

**Métodos:**

##### `build_L_h()`
Construye el operador Laplaciano discreto mediante productos de Kronecker.

**Fundamento Matemático:**

Para la segunda derivada 1D con BC de Robin:
```
D₁ᵤ = (u_{i+1} - 2u_i + u_{i-1}) / h²
```

Con eliminación de punto fantasma para Robin BC:
```
a·u₀ + b·(u₁ - u₋₁)/(2h) = c
→ u₋₁ = u₁ - (2h/b)(c - a·u₀)
```

El operador 2D se construye como:
```
Lₕ = Iᵧ ⊗ Dₓ + Dᵧ ⊗ Iₓ
```

##### `step(u, f=None) → u_new`
Resuelve el sistema lineal para un paso de tiempo.

**Implementación:**
```python
rhs = B.dot(u_vec) + dt * f_vec + dt * q_total * alpha
u_new_vec = _factor(rhs)  # Usa LU prefactorizada
```

**Ejemplo:**
```python
from crank_nicolson import HeatCrankNicolsonSolver
from boundary_conditions import HeatBoundaryCondition

bc = HeatBoundaryCondition(a=0.5, b=1.0, c=12.5)
solver = HeatCrankNicolsonSolver(
    alpha=0.1, dx=0.033, dy=0.033, dt=0.0002,
    nx=30, ny=30, nt=288000,
    robin=bc.to_tuple_x()
)

u = campo_inicial
for paso in range(100):
    u = solver.step(u, fuente_calor)
```

---

### 3. pinn_solver.py

#### Clase: `HeatPINNSolver`

**Descripción:** Utiliza una Red Neuronal Informada por Física preentrenada para la solución.

**Arquitectura PINN:**

```python
class PINN(nn.Module):
    def __init__(self, layers, neurons, activation=nn.Tanh()):
        # Entrada: (x, y, t) → 3 neuronas
        # Ocultas: layers × neurons
        # Salida: u(x,y,t) → 1 neurona
```

**Configuración Estándar:**
- 5 capas ocultas
- 50 neuronas por capa
- Activación Tanh
- Desplazamiento de salida: +25°C (línea base)

**Método Pipeline:**

```python
frames, u_means = HeatPINNSolver.pipeline(
    ibvp, frame, 
    t_steps_per_frame=1,  # no utilizado
    n_frames=20
)
```

**Requisitos Previos:**
- Modelo entrenado en `case3_models/model`
- PyTorch instalado
- CUDA opcional (detección automática de dispositivo)

---

### 4. green_function.py

#### Clase: `GreenFunctionSolver`

**Descripción:** Solución analítica/semi-analítica mediante desarrollo de eigenfunciones.

**Constructor:**
```python
GreenFunctionSolver(alpha, bc, Lx=1.0, Ly=1.0, M=20, N=20)
```

**Fundamento Matemático:**

La solución se representa como:
```
u(x,y,t) = U_amb + Σₘ Σₙ cₘₙ(t) φₘ(x) φₙ(y)
```

**Eigenfunciones φₖ(x):**
```
φₖ(x) = sin(kx) + (k/γ)cos(kx)
donde γ = a/b
```

Estas satisfacen la condición de frontera Robin:
```
a·φₖ + b·φₖ' = 0  en x=0, x=L
```

**Métodos Importantes:**

##### `u(x, y, t, u0_func, f_func=None)`
Método de solución principal.

**Fórmula de Solución:**
```
u(x,y,t) = U_amb + ∫∫ G(x,y,t; x₀,y₀) [u₀(x₀,y₀) - U_amb] dx₀dy₀
           + ∫₀ᵗ ∫∫ G(x,y,t-s; x₀,y₀) f(x₀,y₀,s) dx₀dy₀ ds
```

**Optimizaciones:**
- Las proyecciones se cachean (`_proj_cache`, `_C0`, `_Cf_static`)
- Factores temporales: `exp(-α·L·t)` y `(1 - exp(-α·L·t))/(α·L)`
- Solo cálculo único para f independiente del tiempo

---

### 5. boundary_conditions.py

#### Clase: `HeatBoundaryCondition`

**Descripción:** Gestiona condiciones de frontera para solvers de diferencias finitas.

**Constructor:**
```python
HeatBoundaryCondition(a, b, c)
```

**Tipos de Condiciones de Frontera:**

1. **Dirichlet** (b ≈ 0):
   ```
   u = c/a  en la frontera
   ```

2. **Neumann** (a ≈ 0):
   ```
   ∂u/∂n = c/b  en la frontera
   ```

3. **Robin** (a, b ≠ 0):
   ```
   a·u + b·∂u/∂n = c  en la frontera
   ```

**Método: `apply(u, dx, dy)`**

Aplica condiciones de frontera a los cuatro lados.

**Implementación (Robin):**
```python
# Frontera izquierda (x=0)
u_new[0,:] = (c*dx + b*u[1,:]) / (b + a*dx)

# Frontera derecha (x=L)
u_new[-1,:] = (c*dx + b*u[-2,:]) / (b + a*dx)

# Fronteras inferior/superior análogamente con dy
```

---

### 6. plot_tools.py

#### Funciones

##### `single_plot(u_frame, lx, ly, title, cmap='hot', isolines=False, save_path=None)`

Crea un gráfico de instantánea única.

**Parámetros:**
- `u_frame` (ndarray): Campo de temperatura (ny, nx)
- `lx, ly` (float): Tamaño del dominio
- `title` (str): Título del gráfico
- `cmap` (str): Colormap de Matplotlib
- `isolines` (bool): Dibujar contornos isotérmicos
- `save_path` (str, opcional): Ruta de guardado

**Ejemplo:**
```python
from plot_tools import single_plot

single_plot(
    u_frames[10], 
    lx=1.0, ly=1.0,
    title="Distribución de temperatura en t=30s",
    cmap='hot',
    isolines=True,
    save_path='snapshot_t30.png'
)
```

##### `anim_slide(u_frames, lx, ly, title, cmap='hot', isolines=False)`

Animación interactiva con control deslizante.

**Características:**
- Deslizador para navegación de frames
- Botón Play/Stop para reproducción automática
- Isolíneas dinámicas
- Valores min/max en barra de color

**Ejemplo:**
```python
from plot_tools import anim_slide

anim_slide(
    frames, 
    lx=1.0, ly=1.0,
    title="Propagación de calor en el tiempo",
    cmap='coolwarm',
    isolines=True
)
```

## Ejemplos de Uso

### Ejemplo 1: Comparación Completa de Solvers

```python
from Solver.Python import main

# Ejecuta todos los solvers y crea gráficos de comparación
main()
```

**Salida:**
- Gráfico de residuos PDE
- Gráfico de residuos de condiciones de frontera
- Gráfico de promedios de temperatura
- Imágenes de instantáneas (cada 10 frames)
- Animaciones (Crank-Nicolson, diferencias, errores relativos)

### Ejemplo 2: Problema Personalizado

```python
import numpy as np
from ibvp_data import IBVPData
from frame_data import FrameData
from explicit_solver import HeatExplicitSolver
from plot_tools import anim_slide

# 1. Definir problema
def pulso_gaussiano(x, y):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return 25.0 + 75.0 * np.exp(-r2 / 0.001)

def sin_fuente(x, y, t):
    return np.zeros_like(x)

mi_problema = IBVPData(
    alpha=0.01,
    heat_source=sin_fuente,
    initial_u=pulso_gaussiano,
    a=1.0, b=0.0, c=25.0  # Dirichlet BC: u = 25°C
)

# 2. Configurar malla
mi_malla = FrameData(
    lx=1.0, ly=1.0,
    lt=10.0,
    nx=100, ny=100,
    nt=100000
)

# 3. Ejecutar solver
frames, means = HeatExplicitSolver.pipeline(
    mi_problema, mi_malla,
    t_steps_per_frame=5000,
    n_frames=20,
    use_numba=True
)

# 4. Visualizar
anim_slide(frames, 1.0, 1.0, "Difusión de un pulso gaussiano", cmap='hot')
```

### Ejemplo 3: Análisis de Residuos

```python
from Solver.Python import compute_pde_residual, boundary_residual

# Ejecutar solver
frames, _ = solver.pipeline(ibvp, frame, 1000, 20)

# Calcular residuos PDE
residuos, residuos_medios = compute_pde_residual(frames, frame, ibvp.alpha)

print(f"Residuo PDE máximo: {residuos_medios.max():.6e}")
print(f"Residuo PDE medio: {residuos_medios.mean():.6e}")

# Residuos de condiciones de frontera (último frame)
R_l, R_r, R_b, R_t = boundary_residual(
    frames[-1], frame, 
    k=ibvp.b, h=ibvp.a, u_amb=ibvp.u_amb()
)

print(f"Residuos de frontera (cuadráticos medios):")
print(f"  Izquierda: {R_l.mean():.6e}")
print(f"  Derecha:   {R_r.mean():.6e}")
print(f"  Abajo:     {R_b.mean():.6e}")
print(f"  Arriba:    {R_t.mean():.6e}")
```

## Detalles Matemáticos

### Discretización (Explícita)

**Derivada temporal (diferencia hacia adelante):**
```
∂u/∂t ≈ (u^{n+1} - u^n) / Δt
```

**Operador Laplaciano (diferencias centrales):**
```
∂²u/∂x² ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j}) / Δx²
∂²u/∂y² ≈ (u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / Δy²
```

**Esquema de Actualización:**
```
u^{n+1}_{i,j} = u^n_{i,j} + λₓ(u^n_{i+1,j} - 2u^n_{i,j} + u^n_{i-1,j})
                        + λᵧ(u^n_{i,j+1} - 2u^n_{i,j} + u^n_{i,j-1})
                        + Δt·f_{i,j}
```

donde `λₓ = α·Δt/Δx²` y `λᵧ = α·Δt/Δy²`.

**Condición de Estabilidad (análisis de von Neumann):**
```
λₓ + λᵧ ≤ 1/2
```

### Discretización (Crank-Nicolson)

**Esquema θ:**
```
(u^{n+1} - u^n)/Δt = α[(1-θ)∇²u^n + θ∇²u^{n+1}] + f
```

Para θ = 0.5 (Crank-Nicolson):
```
(u^{n+1} - u^n)/Δt = (α/2)[∇²u^n + ∇²u^{n+1}] + f
```

**Forma Matricial:**
```
(I - θ·Δt·α·Lₕ)u^{n+1} = (I + (1-θ)·Δt·α·Lₕ)u^n + Δt·f
         A                           B
```

**Propiedades:**
- θ = 0: Euler explícito (condicionalmente estable)
- θ = 0.5: Crank-Nicolson (incondicionalmente estable, 2do orden)
- θ = 1: Euler implícito (incondicionalmente estable, 1er orden)

## Optimización de Rendimiento

### Aceleración Numba

**Activación:**
```python
frames, means = HeatExplicitSolver.pipeline(
    ibvp, frame, 1000, 20,
    use_numba=True
)
```

**Ganancias de Velocidad Típicas:**
- Primera ejecución: 1-2s de compilación
- Siguientes: 10-50x más rápido
- Óptimo para nx, ny > 50

### Control de Hilos

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
```

**¿Por qué?**
- Resultados deterministas
- Evitar sobrecarga de CPU
- Mejor rendimiento con múltiples trabajos paralelos

## Extensión

### Agregar Nuevo Solver

**Paso 1: Crear Clase de Solver**

```python
# my_solver.py
class MiSolverPersonalizado:
    def __init__(self, alpha, dx, dy, dt, bc):
        self.alpha = alpha
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.apply_bc = bc
    
    def step(self, u, f=None):
        # Implementar un paso de tiempo
        u_new = ... # Su método aquí
        return u_new
    
    @staticmethod
    def pipeline(ibvp, frame, t_steps_per_frame, n_frames):
        # Simulación completa
        # ... código de configuración ...
        
        solver = MiSolverPersonalizado(alpha, dx, dy, dt, bc)
        frames = [u0]
        u_means = []
        
        for n in range(n_frames):
            u = solver.n_steps(u, f, t_steps_per_frame)
            frames.append(u.copy())
            u_means.append(u.mean())
            # Registro ...
        
        return frames, u_means
```

**Paso 2: Integración en Comparación**

```python
# En Solver.Python.py
from my_solver import MiSolverPersonalizado

data = {
    "Mi Método": CaseData(MiSolverPersonalizado.pipeline, "-", "#ff7f00", 'v'),
    # ... otros solvers ...
}
```

## Referencias y Lecturas Adicionales

### Literatura

1. **Diferencias Finitas:**
   - LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*
   
2. **Crank-Nicolson:**
   - Crank, J., & Nicolson, P. (1947). *A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type*

3. **PINNs:**
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems*

### Recursos en Línea

- [Documentación NumPy](https://numpy.org/doc/)
- [Matrices Dispersas SciPy](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Guía de Usuario Numba](https://numba.readthedocs.io/)
- [Tutoriales PyTorch](https://pytorch.org/tutorials/)

---

## Licencia y Contacto

Este proyecto es código abierto bajo la Licencia MIT.

**Repositorio:** https://github.com/Haasrobertgmxnet/HeatConduction

**Issues y contribuciones** son bienvenidos!