from dataclasses import dataclass

@dataclass
class FrameData:
    """
    Stores spatial and temporal discretization settings for the simulation frame.

    Parameters:
        lx (float): Total length of the domain in x-direction.
        ly (float): Total length of the domain in y-direction.
        lt (float): Total simulated time duration.
        nx (int): Number of grid points along x.
        ny (int): Number of grid points along y.
        nt (int): Number of time steps.
    """

    lx: float  # Domain size in x-direction
    ly: float  # Domain size in y-direction
    lt: float  # Total simulation time
    nx: int    # Number of x-grid nodes
    ny: int    # Number of y-grid nodes
    nt: int    # Number of time steps

    def dx(self) -> float:
        """Return spatial grid spacing in x-direction."""
        return self.lx / (self.nx - 1)

    def dy(self) -> float:
        """Return spatial grid spacing in y-direction."""
        return self.ly / (self.ny - 1)

    def dt(self) -> float:
        """Return time-step size."""
        return self.lt / (self.nt - 1)

# Example configurations

# Coarse grid
# frame1 = FrameData(1.0, 1.0, 60.0, 30, 30, 288000)

# Medium grid
frame1 = FrameData(1.0, 1.0, 60.0, 60, 60, 288000)

# Tall grid (anisotropic resolution)
frame2 = FrameData(1.0, 1.0, 60.0, 30, 300, 288000)

# More refined resolution options (commented)
# frame1 = FrameData(1.0, 1.0, 60.0, 90, 90, 288000)
# frame1 = FrameData(1.0, 1.0, 60.0, 120, 120, 4 * 288000)
