import math
import torch
import torch.nn as nn
from dataclasses import dataclass

try:
    from training_phase_config import TrainingPhaseConfig
except:
    from heat_pinn_python.training_phase_config import TrainingPhaseConfig

class TransformTemp:
    def __init__(self, A: float=1, B:float = 0):
        self.A = A
        self.B = B

    def scale(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.B) / self.A

    def inv_scale(self, scaled_u: torch.Tensor) -> torch.Tensor:
        return self.A * scaled_u + self.B

    def scale_heat_source(self, f: torch.Tensor) -> torch.Tensor:
        return f/self.A

    def scale_bc_c(self, a: float, c: float) -> float:
        return c / self.A +a * self.B

    def scale_single(self, u: float) -> float:
        return (u - self.B) / self.A

temp_transform = TransformTemp(A=300, B=0)

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

    def prepare_input(*inputs):
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

    def fourier_features(self, xyt: torch.Tensor) -> torch.Tensor:
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
        global temp_transform
        xyt = PINN.prepare_input(*inputs)  # (N,3)

        if self.use_fourier:
            h = self.fourier_features(xyt)
        else:
            h = xyt

        for layer in self.layers:
            h = self.activation(layer(h))

        u = self.out_layer(h)  # (N,1)
        u = u + temp_transform.scale_single(25.0)
        return u

    @torch.no_grad()
    def predict_u(model, xyt: torch.Tensor) -> torch.Tensor:
        """
        Schnelle Vorhersage nur von u(x,y,t).
        Kein Autograd notwendig, ideal für Animationen und Auswertung.
        """
        return model(xyt)[:, 0:1]

    def predict_u_and_derivs(model, xyt: torch.Tensor):
    # Erzwinge Gradienten
        xyt = xyt.clone().detach().requires_grad_(True)

        u = model(xyt)

        grads = torch.autograd.grad(
            outputs=u,
            inputs=xyt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=xyt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=xyt,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 1:2]

        lap_u = u_xx + u_yy
        return u, u_t, lap_u

@dataclass
class PINNConfig:
    n_hid_layers: int = 5
    n_neurons: int = 50
    use_fourier = False
    m_fourier: int = 12
    fourier_scale: float = 2.0

def load_model(model_path: str, pinn_cfg: PINNConfig, device: str="cpu"):
    model = PINN(
        hid_layers=pinn_cfg.n_hid_layers,
        neurons=pinn_cfg.n_neurons,
        activation=nn.Tanh(),
        use_fourier=pinn_cfg.use_fourier,
        m_fourier=pinn_cfg.m_fourier,
        fourier_scale=pinn_cfg.fourier_scale
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only = True)
    model.load_state_dict(checkpoint["model_state"])   
    model.eval()

    print(f"Modell geladen aus {model_path}")
    return model, checkpoint
