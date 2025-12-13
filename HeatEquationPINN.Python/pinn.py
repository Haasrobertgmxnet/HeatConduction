import math
import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    PINN mit optionalen Fourier-Features.

    use_fourier = False: klassisches MLP mit Input (x,y,t)
    use_fourier = True:  Random Fourier Features auf (x,y,t), dann MLP
    """

    def __init__(
        self,
        hid_layers: int,
        neurons: int,
        activation: nn.Module = nn.Tanh(),
        use_fourier: bool = False,
        m_fourier: int = 40,
        fourier_scale: float = 5.0,
    ):
        super().__init__()
        self.activation = activation
        self.use_fourier = use_fourier

        if use_fourier:
            # Eingabedimension 3 (x,y,t) → projiziere in R^{m_fourier}
            # und bilde sin/cos → 2*m Fourier-Features
            self.m_fourier = m_fourier
            self.fourier_scale = fourier_scale

            # feste (nicht trainierbare) Random-Matrix B
            B = torch.randn(3, m_fourier) * fourier_scale
            self.register_buffer("B", B)   # B ist Teil des Modells, aber ohne Gradienten

            in_dim = 2 * m_fourier
        else:
            self.m_fourier = 0
            self.fourier_scale = 0.0
            self.register_buffer("B", torch.zeros(3, 1))  # Dummy
            in_dim = 3  # (x,y,t)

        # MLP
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, neurons))
        for j in range(hid_layers):
            print( f"Adding layer {j+1} with {neurons} neurons." )
            self.layers.append(nn.Linear(neurons, neurons))

        self.out_layer = nn.Linear(neurons, 1)  # EIN Output: u


    def _prepare_input(self, *inputs):
        """
        Erlaubt forward(x,y,t) oder forward(xyt)
        und gibt ein Tensor (N,3) zurück.
        """
        if len(inputs) == 3:
            x, y, t = inputs
            xyt = torch.cat([x, y, t], dim=1)
        elif len(inputs) == 1:
            xyt = inputs[0]
        else:
            raise ValueError("Expected (x,y,t) or (xyt) as input.")
        return xyt


    def _fourier_features(self, xyt: torch.Tensor) -> torch.Tensor:
        """
        Random Fourier Features aus (x,y,t).
        xyt: (N,3)
        Rückgabe: (N, 2*m_fourier)
        """
        # (N,3) @ (3,m) = (N,m)
        proj = 2.0 * math.pi * xyt @ self.B   # skaliert
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)


    def forward(self, *inputs):
        """
        forward(x,y,t) oder forward(xyt)
        → u(x,y,t) als (N,1)
        """
        xyt = self._prepare_input(*inputs)  # (N,3)

        if self.use_fourier:
            h = self._fourier_features(xyt)
        else:
            h = xyt

        for layer in self.layers:
            h = self.activation(layer(h))

        u = self.out_layer(h)  # (N,1)
        u = u + 25.0
        return u
