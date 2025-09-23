import torch
import torch.nn as nn
import copy
import time

device = torch.device("cuda" if torch.cuda.is_available () else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def generate_data(n_points, length, total_time, seed=None, device='cpu'):
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.rand(n_points, 1, requires_grad=True) * length
    y = torch.rand(n_points, 1, requires_grad=True) * length
    t = torch.rand(n_points, 1, requires_grad=True) * total_time
    
    n_boundary = n_points // 10
    x_boundary = torch.cat([torch.zeros(n_boundary, 1), torch.full((n_boundary, 1), length)], dim=0)
    y_boundary = torch.cat([torch.zeros(n_boundary, 1), torch.full((n_boundary, 1), length)], dim=0)
    t_boundary = torch.rand(2 * n_boundary, 1, requires_grad=True) * total_time
    
    x = torch.cat([x, x_boundary, torch.rand(2 * n_boundary, 1) * length], dim=0)
    y = torch.cat([y, torch.rand(2 * n_boundary, 1) * length, y_boundary], dim=0)
    t = torch.cat([t, t_boundary, t_boundary], dim=0)
    
    return x.to(device), y.to(device), t.to(device)

# SHOULD BE the pipeline of Aryal
def pipeline1():
    set_seed(0)
    class PINN(nn.Module):
        def __init__(self , layers , neurons , activation=nn.Tanh()):
            super(PINN , self).__init__ ()
            self.activation = activation
            self.layers = nn.ModuleList ()
            self.layers.append(nn.Linear(3, neurons))
            for _ in range(layers - 1):
                self.layers.append(nn.Linear(neurons , neurons))
            self.layers.append(nn.Linear(neurons , 1))
        def forward(self , x, y, t):
            inputs = torch.cat([x, y, t], dim=1)
            output = inputs
            for layer in self.layers[:-1]:
                output = self.activation(layer(output))
            output = self.layers[-1](output) + 25
            return output

    def pde_loss(model , x, y, t, epsilon , f):
        u = model(x, y, t)

        # u_t wird nicht weiter abgeleitet False
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # u_x und u_y werden weiter abgeleitet True
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # zweite Ableitung braucht keinen neuen Graphen False
        u_xx = torch.autograd.grad(u_x , x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y , y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        residual = u_t - epsilon * (u_xx + u_yy) - f
        return torch.mean(residual ** 2)

    def initial_loss(model , x, y, t, u0):
        u = model(x, y, t)
        return torch.mean((u - u0) ** 2)

    def boundary_loss(model, x, y, t, length):
        u = model(x, y, t)
        x_boundary = (x <= 1e-6) | (x >= length - 1e-6)
        y_boundary = (y <= 1e-6) | (y >= length - 1e-6)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
        create_graph=True)[0]
        loss_x = torch.mean(u_x[x_boundary] ** 2)
        loss_y = torch.mean(u_y[y_boundary] ** 2)
        return loss_x + loss_y

    def heat_source(x, y, center_x, center_y, radius, strength, t):
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        return torch.where(t > 0, strength * torch.exp(-distance**2 / (2 * radius**2)), torch.zeros_like(t))

    # Data from Aryal's thesis

    ## Learning data
    leraning_rate = 1e-3
    epochs = 3000
    
    ## NN Architecture
    hid_layers = 5
    nodes = 50

    ## Physics data
    epsilon = 1e-1 # alpha
    heat_radius = 0.1
    heat_strength = 500

    ## Geometry of squared domain
    length = length_x = length_y = 1.0

    ## weights of residuals
    weight_physics = 1.0
    weight_initial = 1.0
    weight_boundary = 1.0

    ### my value
    ## weight_physics = 20.0

    model = PINN(hid_layers,nodes).to(device)
    model1a= copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters (), lr= leraning_rate)
        
    n_points= 1000
    total_time = 5.0

    start = time.time()
    for epoch in range(epochs):
        x, y, t = generate_data(n_points , length , total_time, epoch)
        u0 = torch.full_like(x, 25) # Initial condition (room temperature)

        
        f = heat_source(x, y, 0.5*length, 0.5*length , heat_radius, heat_strength , t)
        optimizer.zero_grad ()
        loss_physics = pde_loss(model , x, y, t, epsilon , f)
        loss_initial = initial_loss(model , x, y, torch.zeros_like(t), u0)
        loss_boundary = boundary_loss(model , x, y, t, length)
        loss = weight_physics * loss_physics + weight_initial * loss_initial + weight_boundary * loss_boundary
        loss.backward ()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:5d}: total={loss.item():.6e}, "
                  f"phy={loss_physics.item():.6e}, ic={loss_initial.item():.6e}, bc={loss_boundary.item():.6e}")

    print(f"Training finished and took {time.time()-start:.4} seconds.")
    model1 = copy.deepcopy(model)
    return model1, model1a

# IS the pipeline of Aryal
def pipeline2():
    set_seed(0)
    # Define the PINN model
    class PINN(nn.Module):
        def __init__(self, layers, neurons, activation=nn.Tanh()):
            super(PINN, self).__init__()
            self.activation = activation
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(3, neurons))
            for _ in range(layers - 1):
                self.layers.append(nn.Linear(neurons, neurons))
            self.layers.append(nn.Linear(neurons, 1))

        def forward(self, x, y, t):
            inputs = torch.cat([x, y, t], dim=1)
            output = inputs
            for layer in self.layers[:-1]:
                output = self.activation(layer(output))
            output = self.layers[-1](output) + 25  # Add 25°C to the output to start from room temperature
            return output

    # Define the loss functions
    def pde_loss(model, x, y, t, epsilon, f):
        u = model(x, y, t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
        residual = u_t - epsilon * (u_xx + u_yy) - f
        return torch.mean(residual ** 2)

    def initial_loss(model, x, y, t, u0):
        u = model(x, y, t)
        return torch.mean((u - u0) ** 2)

    def boundary_loss(model, x, y, t, length):
        u = model(x, y, t)
    
        x_boundary = (x <= 1e-6) | (x >= length - 1e-6)
        y_boundary = (y <= 1e-6) | (y >= length - 1e-6)
    
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
        loss_x = torch.mean(u_x[x_boundary]**2)
        loss_y = torch.mean(u_y[y_boundary]**2)
    
        return loss_x + loss_y

    

    def heat_source(x, y, center_x, center_y, radius, strength, t):
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        return torch.where(t > 0, strength * torch.exp(-distance**2 / (2 * radius**2)), torch.zeros_like(t))

    # Simulation parameters
    length = 1.0
    total_time = 5.0
    n_points = 1000
    n_points_plot = 100
    layers = 5
    neurons = 50
    epochs = 3000
    learning_rate = 0.001
    epsilon = 0.1

    # Initialize model and optimizer
    model = PINN(layers, neurons).to(device)
    model2a= copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight_residual = 1.0
    weight_initial = 1.0
    weight_boundary = 1.0

    # Heat source parameters
    center_x, center_y = 0.5, 0.5
    heat_radius = 0.1
    heat_strength = 500.0  # Increased heat strength to reach higher temperatures

    # Training loop
    start = time.time()
    for epoch in range(epochs):
        x, y, t = generate_data(n_points, length, total_time, epoch)
        u0 = torch.full_like(x, 25)  # Initial condition (room temperature)
        f = heat_source(x, y, center_x, center_y, heat_radius, heat_strength, t)

        optimizer.zero_grad()
        loss_residual = pde_loss(model, x, y, t, epsilon, f)
        loss_initial = initial_loss(model, x, y, torch.zeros_like(t), u0)
        loss_boundary = boundary_loss(model, x, y, t, length)
        loss = weight_residual * loss_residual + weight_initial * loss_initial + weight_boundary * loss_boundary
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:5d}: total={loss.item():.6e}, "
                      f"phy={loss_residual.item():.6e}, ic={loss_initial.item():.6e}, bc={loss_boundary.item():.6e}")
            # print(f'Epoch {epoch}, Loss: {loss.item()}')

    print(f"Training finished and took {time.time()-start:.4} seconds.")
    model2 = copy.deepcopy(model)
    return model2, model2a

def exec_comparison():
    model1, model1a = pipeline1()
    model2, model2a = pipeline2()

    print("UNTRAINED MODELS")
    sd1 = model1a.state_dict()
    sd2 = model2a.state_dict()

    for name, param1 in sd1.items():
        param2 = sd2[name]
        diff = (param1 - param2).abs().max()
        print(f"{name}: max abs diff = {diff.item()}")

    print("TRAINED MODELS")
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    for name, param1 in sd1.items():
        param2 = sd2[name]
        diff = (param1 - param2).abs().max()
        print(f"{name}: max abs diff = {diff.item()}")