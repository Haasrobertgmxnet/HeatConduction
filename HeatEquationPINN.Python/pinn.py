import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Fully-connected feedforward neural network that approximates u(x, y, t).

    Architecture:
    - Input dimension = 3 (x, y, t)
    - Several hidden layers of width `neurons` using activation function
    - Output is a single scalar value u(x, y, t)
    """
    def __init__(self , layers , neurons , activation=nn.Tanh()):
        print("PINN __init__ called")
        super(PINN , self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3, neurons))  # input layer: (x,y,t)
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(neurons , neurons))  # hidden layers
        self.layers.append(nn.Linear(neurons , 1))  # final output layer

    def forward(self, *inputs):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        *inputs: ptr to tensor of shape either (N,3), or (N, 1)
            Coordinates to evaluate the network at.

        Returns
        -------
        u : tensor of shape (N, 1)
            Predicted temperature at the given points.
        """
        if len(inputs) == 3:
            x, y, t = inputs
            xyt = torch.cat([x, y, t], dim=1)
        elif len(inputs) == 1:
            xyt = inputs[0]
        else:
            raise ValueError("Expected (x,y,t) or (xyt) as input.")

        out = xyt
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))

        out = self.layers[-1](out)

        return out + 25.0

    def __forward(self, x, y, t):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x, y, t : tensor of shape (N, 1)
            Coordinates to evaluate the network at.

        Returns
        -------
        u : tensor of shape (N, 1)
            Predicted temperature at the given points.
        """
        inputs = torch.cat([x, y, t], dim=1)
        output = inputs
        for layer in self.layers[:-1]:
            output = self.activation(layer(output))
        output = self.layers[-1](output) + 25  # shift baseline temperature
        return output
