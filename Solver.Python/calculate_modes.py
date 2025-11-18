import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

def func(z):
    gamma = 0.5
    return z*z*np.sin(z)-2*z*gamma*np.cos(z) - gamma*gamma*np.sin(z)

def solve(func, cutoff=100, coarse_points=10_000, refine=False, refine_points=1_000, x_min=0, x_max=500):

    # Coarse grid search
    z_vals = np.linspace(x_min, x_max, num=coarse_points)
    f_vals = func(z_vals)

    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    roots = []

    for idx in sign_changes:
        z1, z2 = z_vals[idx], z_vals[idx + 1]

        if refine:
            # optionales lokales Refinement
            zr = np.linspace(z1, z2, refine_points)
            fr = func(zr)
            sc = np.where(np.diff(np.sign(fr)))[0]
            if len(sc) > 0:
                z1, z2 = zr[sc[0]], zr[sc[0] + 1]

        try:
            sol = root_scalar(func, bracket=[z1, z2],
                                method='brentq',
                                xtol=1e-12, rtol=1e-12)
            roots.append(sol.root)
        except ValueError:
            pass

        if len(roots) >= cutoff:
            break

    return np.array(roots)

def fplot(_cutoff=500):
    

    def solve0(func, cutoff=_cutoff, coarse_points=10000, refine_points=1000):
        return solve(func, refine=True, refine_points=1000)

    def solve1(func, cutoff = _cutoff):
        return solve(func, refine=False)
        
    roots0 = solve0(func, _cutoff)
    roots1 = solve1(func, _cutoff)

    print(f"Roots 0: {roots0}")
    print(f"Roots 1: {roots1}")


    # --- Plot ---
    upper_limit = 20
    z = np.linspace(0.0,upper_limit, num=200)
    plt.figure(figsize=(10, 5))
    plt.plot(z, func(z), 'g--', linewidth=1, markersize=3, label='Function $ F(\mu )$')
    plt.axhline(0, color='black', lw=1)

    # Nullstellen markieren
    plt.scatter(roots1, func(roots1), color='red', zorder=5, s=47, label='Zeros of $ F(\mu )$')
    for r in roots1[:7]:
        plt.text(r, func(r) + 0.1, f"{r:.4f}", 
                    ha='left', va='bottom', fontsize=12, rotation=45, color='blue')

    ref = 0.96018887
    vals = []
    for ele0 in [-5e-6, -4e-6, -3e-6, -2e-6, -1e-6, 0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6]:
        ele = ref + ele0
        vals.append(func(ele))
        print(f"func at {ele:.8f}: {func(ele):.8f}")

    print(np.min(np.abs(vals)))

    # plt.title('Function $ F(\mu ) = (\gamma^2  - \mu^2) \mathrm{sin}(\mu ) + 2 \mu \gamma \mathrm{cos }(\mu )$ and its Zeros')
    plt.xlabel('$ \mu $')
    plt.ylabel('$ F(\mu )$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, upper_limit)
    plt.show()

