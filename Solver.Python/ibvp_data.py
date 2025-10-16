class IBVPData:
    def __init__(self, alpha, heat_source, initial_u, a, b, c):
        self.alpha = alpha
        self.heat_source = heat_source
        self.initial_u = initial_u
        self.a = a
        self.b = b
        self.c = c

from function_set import GaussKernel, ConstantFunc

gauss_kernel = GaussKernel(0.5, 0.5, 0.1, 500.0)
heat_source = gauss_kernel.evaluate

constant_f = ConstantFunc(25.0)
initial_u = constant_f.evaluate

ibvp1 = IBVPData(0.1, heat_source, initial_u, 0.5, 1, 12.5)
# ibvp1 = IBVPData(0.1, heat_source, initial_u, 0, 1, 0)