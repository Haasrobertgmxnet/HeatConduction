class HeatBoundaryCondition:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def apply_robin(self, u, dx, dy):
        u_new = u.copy()
        if self.b == 0:  # Dirichlet
            u_new[0,:] = u_new[-1,:] = u_new[:,0] = u_new[:,-1] = self.c / self.a  # falls a<>0
            return u_new
        if self.a == 0: # Neumann
            u_new[0,:] = self.c*dx/self.b + u[1,:]
            u_new[-1,:] = self.c*dx/self.b + u[-2,:]

            u_new[:,0] = self.c*dy/self.b + u[:,1]
            u_new[:,-1] = self.c*dy/self.b + u[:,-2]

            return u_new
            u_new[0,:] = (self.c*dx + self.b*u[1,:]) / self.b
            u_new[-1,:] = - (self.c*dx - self.b*u[-2,:]) / self.b

            u_new[:,0] = (self.c*dy + self.b*u[:,1]) /  self.b
            u_new[:,-1] = -(self.c*dy - self.b*u[:,-2]) /  self.b
            return u_new

        else:  # Robin
            u_new[0,:] = (self.c*dx + self.b*u[1,:]) / (self.b + self.a*dx)
            u_new[-1,:] = (self.c*dx + self.b*u[-2,:]) / (self.b + self.a*dx)

            u_new[:,0] = (self.c*dy + self.b*u[:,1]) / (self.b + self.a*dy)
            u_new[:,-1] = (self.c*dy + self.b*u[:,-2]) / (self.b + self.a*dy)

            return u_new
        u_new[0,:] = (self.c*dx + self.b*u[1,:]) / (self.b + self.a*dx)
        u_new[-1,:] = (self.c*dx - self.b*u[-2,:]) / (-self.b + self.a*dx)

        u_new[:,0] = (self.c*dy + self.b*u[:,1]) / (self.b + self.a*dy)
        u_new[:,-1] = (self.c*dy - self.b*u[:,-2]) / (-self.b + self.a*dy)

        return u_new
