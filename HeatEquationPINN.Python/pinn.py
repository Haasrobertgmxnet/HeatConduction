import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Fully-connected feedforward neural network that approximates:
      u(x,y,t), v(x,y,t)=u_t, w(x,y,t)=u_xx+u_yy

    Input:  (x,y,t) ∈ R^3
    Output: (u,v,w) ∈ R^3
    """
    def __init__(self, layers, neurons, activation=nn.Tanh()):
        super(PINN, self).__init__()
        self.activation = activation

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(3, neurons))
        # Hidden layers
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(neurons, neurons))
        # Output layer: 3 Kanäle (u,v,w)
        self.out_layer = nn.Linear(neurons, 3)

    def forward(self, *inputs):
        """
        forward(x, y, t) oder forward(xyt)
        -> Tensor (N,3): [u, v, w]
        """
        if len(inputs) == 3:
            x, y, t = inputs
            xyt = torch.cat([x, y, t], dim=1)
        elif len(inputs) == 1:
            xyt = inputs[0]
        else:
            raise ValueError("Expected (x,y,t) or (xyt) as input.")

        h = xyt
        for layer in self.layers:
            h = self.activation(layer(h))

        out = self.out_layer(h)  # (N,3)

        # nur u bekommt den +25 Offset
        u = out[:, 0:1] + 25.0
        vw = out[:, 1:]
        return torch.cat([u, vw], dim=1)  # (N,3)
