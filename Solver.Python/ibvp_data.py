class IBVPData:
    """
    Stores data for an initial boundary value problem (IBVP) for the heat equation.
    
    Parameters:
        alpha (float): Heat transfer coefficient (thermal diffusivity).
        heat_source (callable): Function f(x, y, t) defining the heat source term.
        initial_u (callable): Function u0(x, y) defining the initial temperature field.
        a, b, c (float): Parameters for the Robin boundary condition:
                         a * u + b * (∂u/∂n) = c on the boundary.
    """

    def __init__(self, alpha: float, heat_source, initial_u, a: float, b: float, c: float):
        self.alpha = alpha
        self.heat_source = heat_source
        self.initial_u = initial_u
        self.a = a
        self.b = b
        self.c = c

    def u_amb(self):
        """
        Returns the ambient (equilibrium) temperature that follows from the
        Robin boundary condition. If a == 0, the boundary condition does not
        define such a value, so None is returned.

        For a ≠ 0:
            u_amb = c / a
        """
        if self.a == 0:
            return None
        return self.c / self.a


# Example usage ---------------------------------------------------------------

from function_set import GaussKernel, CompactGaussKernel, CompactCylindricalKernel, ModeFunction, ConstantFunc

# Choose the heat source function via kernel
gauss_kernel = GaussKernel(0.5, 0.5, 0.1, 500.0)
heat_source = gauss_kernel.evaluate

# Choose the heat source function via kernel
gauss_kernel = GaussKernel(0.5, 0.5, 0.1, 500.0)
heat_source = gauss_kernel.evaluate

f = ModeFunction()
# heat_source = f.evaluate

# Initial temperature distribution
initial_temp = ConstantFunc(25.0)
initial_u = initial_temp.evaluate

# Create IBVP configuration with a Robin boundary condition
ibvp1 = IBVPData(
    alpha=0.1,
    heat_source=heat_source,
    initial_u=initial_u,
    a=0.5,
    b=1.0,
    c=12.5
)

# Print ambient temperature if meaningful
print("Ambient temperature:", ibvp1.u_amb())
