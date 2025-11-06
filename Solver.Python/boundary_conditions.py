class HeatBoundaryCondition:
    def __init__(self, a, b, c):
        # Store the boundary condition coefficients
        # a, b, c correspond to coefficients in the boundary condition equation:
        # a * u + b * (du/dn) = c
        self.a = a
        self.b = b
        self.c = c

    def apply(self, u, dx, dy):
        # Apply the boundary conditions to the 2D temperature field u
        # dx and dy are the grid spacings in the x and y directions
        u_new = u.copy()
        tiny = 1e-14

        if abs(self.b) < tiny:  # Dirichlet boundary condition
            # a * u = c  →  u = c / a
            # Apply constant boundary value along all domain boundaries
            u_new[0,:] = u_new[-1,:] = u_new[:,0] = u_new[:,-1] = self.c / self.a

        if abs(self.a) < tiny: # Neumann boundary condition
            # Neumann sets the derivative normal to the boundary
            # u(boundary) = u(adjacent cell) + (c * dx) / b
            u_new[0,:] = self.c*dx/self.b + u[1,:]
            u_new[-1,:] = self.c*dx/self.b + u[-2,:]

            u_new[:,0] = self.c*dy/self.b + u[:,1]
            u_new[:,-1] = self.c*dy/self.b + u[:, -2]

        else:  # Robin boundary condition
            # a * u + b * (du/dn) = c solved for u(boundary)
            # This blends Dirichlet and Neumann behaviors
            u_new[0,:] = (self.c*dx + self.b*u[1,:]) / (self.b + self.a*dx)
            u_new[-1,:] = (self.c*dx + self.b*u[-2,:]) / (self.b + self.a*dx)

            u_new[:,0] = (self.c*dy + self.b*u[:,1]) / (self.b + self.a*dy)
            u_new[:,-1] = (self.c*dy + self.b*u[:, -2]) / (self.b + self.a*dy)

        return u_new

    def to_tuple_x(self):
        # Return coefficients for x direction boundary conditions
        return (self.a, self.b, self.c,
                self.a, self.b, self.c)

    def to_tuple_y(self):
        # Return coefficients for y direction boundary conditions
        return (self.a, self.b, self.c,
                self.a, self.b, self.c)
