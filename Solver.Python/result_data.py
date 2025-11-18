from dataclasses import dataclass

@dataclass
class result_data:
    """
    Stores result data of calculated by the (static) pipeline method of a solver class.
    """
    def __init__ (self, u, u_t= None, u_x= None, u_y= None, u_xx= None, u_yy= None):
        self.u = u
        self.u_t = u_t
        self.u_x = u_x
        self.u_y = u_y
        self.u_xx = u_xx
        self.u_yy = u_yy
        if u_xx is not None and u_yy is not None:
            self.laplacian = -(u_xx + u_yy)
        else:
            self.laplacian = None





