from dataclasses import dataclass
from typing import Optional

@dataclass
class result_data:
    """
    Stores result data calculated by the (static) pipeline method of a solver class.
    """
    u: any
    u_t: Optional[any] = None
    u_x: Optional[any] = None
    u_y: Optional[any] = None
    u_xx: Optional[any] = None
    u_yy: Optional[any] = None
    lap_u: Optional[any] = None

    def __post_init__(self):
        has_second_derivatives = self.u_xx is not None and self.u_yy is not None
        has_laplacian = self.lap_u is not None

        # XOR-Check
        if (has_second_derivatives and has_laplacian):
            raise ValueError(
                "Either provide u_xx and u_yy OR lap_u, but not both."
            )

        if has_second_derivatives:
            self.laplacian = -(self.u_xx + self.u_yy)
            return
        if has_laplacian:
            self.laplacian = self.lap_u
            return
        self.laplacian = None
