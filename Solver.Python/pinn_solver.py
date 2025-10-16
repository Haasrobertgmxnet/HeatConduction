import torch
import torch.nn as nn
import copy
import time
import numpy as np
from comparison import compare_trained

class HeatPINNSolver():
    def __init__(self):
        pass
    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
        print("PINN solver")
        class PINN(nn.Module):
            def __init__(self , layers , neurons , activation=nn.Tanh()):
                super(PINN , self).__init__ ()
                self.activation = activation
                self.layers = nn.ModuleList ()
                self.layers.append(nn.Linear(3, neurons))
                for _ in range(layers - 1):
                    self.layers.append(nn.Linear(neurons , neurons))
                self.layers.append(nn.Linear(neurons , 1))
            def forward(self, x, y, t):
                inputs = torch.cat([x, y, t], dim=1)
                output = inputs
                for layer in self.layers[:-1]:
                    output = self.activation(layer(output))
                output = self.layers[-1](output) + 25
                return output
        
        ## NN Architecture
        hid_layers = 5
        nodes = 50
        model = PINN(hid_layers,nodes).to(device)
        model.load_state_dict(torch.load('case3_models/model'))
        model.eval()

        print("Modell erfolgreich geladen.")
        x_vis = torch.linspace(0.0, frame.lx, frame.nx)
        y_vis = torch.linspace(0.0, frame.ly, frame.ny)
        t_vis = torch.linspace(0.0, frame.lt, frame.nt)
        Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')
        Xv = Xv.flatten()
        Yv = Yv.flatten()
        u0 =ibvp.initial_u(Xv, Yv).detach().cpu().reshape(frame.ny, frame.nx).numpy()

        u_frames = [u0,  ] 
        u_means = []
        with torch.no_grad():
            for n_frame in range(n_frames):

                tval = frame.lt*(1+n_frame)/n_frames
                xv = Xv.unsqueeze(1)  # (N,1)
                yv = Yv.unsqueeze(1)  # (N,1)
                tv = torch.full_like(Xv, tval).unsqueeze(1)  # (N,1)
                u_pred = model(xv, yv, tv).reshape(frame.ny, frame.nx).cpu().numpy()
                # u_pred = model(Xv.flatten(), Yv.flatten(), torch.full_like(Xv.flatten(), tval)).reshape(frame.ny, frame.nx).cpu().numpy()
                u_frames.append(u_pred)
                u_mean = u_pred.mean()
                u_means.append(u_mean)
                print(f"Frame {tval:.2f}: u mean={u_mean:.6f}, ")

        return u_frames, u_means

#### OLD:

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

    def pipeline__(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
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
            def forward(self, x, y, t):
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
            squared_distance = (x - center_x)**2 + (y - center_y)**2
            return torch.where(t > 0, strength * torch.exp(-squared_distance / (2 * radius**2)), torch.zeros_like(t))

        # Data from Aryal's thesis

        ## Learning data
        leraning_rate = 1e-3
        epochs = 3000
    
        ## NN Architecture
        hid_layers = 5
        nodes = 50

        ## Physics data
        epsilon = ibvp.alpha

        ## Geometry of squared domain
        length = length_x = length_y = frame.lx

        ## weights of residuals
        weight_physics = 1.0
        weight_initial = 1.0
        weight_boundary = 1.0

        model = PINN(hid_layers,nodes).to(device)
        optimizer = torch.optim.Adam(model.parameters (), lr= leraning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=4000,
            gamma=0.5
        )
        
        n_points= 1000
        total_time = frame.lt

        # pre config early stopping
        best_loss = float('inf')     # bester bisheriger Verlust
        patience = 500               # Anzahl Epochen ohne Verbesserung, bevor wir abbrechen
        trigger_times = 0  

        start = time.time()
        for epoch in range(epochs):
            x, y, t = generate_data(n_points , length , total_time, epoch)
            u0 = torch.full_like(x, 25) # Initial condition (room temperature)

            f = ibvp.heat_source(x,y,t)
            optimizer.zero_grad ()
            loss_physics = pde_loss(model , x, y, t, epsilon , f)
            loss_initial = initial_loss(model , x, y, torch.zeros_like(t), u0)
            loss_boundary = boundary_loss(model , x, y, t, length)
            loss = weight_physics * loss_physics + weight_initial * loss_initial + weight_boundary * loss_boundary
            loss.backward ()
            optimizer.step()
            # scheduler.step()

            current_loss = loss.item()

            if current_loss < best_loss - 1e-6:  # 1e-6 = kleine Toleranz
                best_loss = current_loss
                trigger_times = 0
                # Optional: bestes Modell speichern
                torch.save(model.state_dict(), "best_model.pt")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Triggered early stopping after {epoch} epochs!")
                    # Lade bestes Modell zurück
                    model.load_state_dict(torch.load("best_model.pt"))
                    # break

            if epoch % 100 == 0 or epoch == epochs-1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d}: total={loss.item():.6e}, "
                      f"phy={loss_physics.item():.6e}, ic={loss_initial.item():.6e}, bc={loss_boundary.item():.6e}, lr={current_lr:.6e}")

        print(f"Training finished and took {time.time()-start:.4} seconds.")
        print("Comparison with reference model:")
        compare_trained(model)
    
        x_vis = torch.linspace(0.0, frame.lx, frame.nx)
        y_vis = torch.linspace(0.0, frame.ly, frame.ny)
        t_vis = torch.linspace(0.0, frame.lt, frame.nt)
        Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')
        Xv = Xv.flatten()
        Yv = Yv.flatten()
        u0 =ibvp.initial_u(Xv, Yv).detach().cpu().reshape(frame.ny, frame.nx).numpy()

        u_frames = [u0,  ] 
        u_means = []
        with torch.no_grad():
            for _ in range(n_frames):
                tval = t_steps_per_frame*frame.lt//frame.nt
                xv = Xv.unsqueeze(1)  # (N,1)
                yv = Yv.unsqueeze(1)  # (N,1)
                tv = torch.full_like(Xv, tval).unsqueeze(1)  # (N,1)
                u_pred = model(xv, yv, tv).reshape(frame.ny, frame.nx).cpu().numpy()
                # u_pred = model(Xv.flatten(), Yv.flatten(), torch.full_like(Xv.flatten(), tval)).reshape(frame.ny, frame.nx).cpu().numpy()
                u_frames.append(u_pred)
                u_mean = u_pred.mean()
                u_means.append(u_mean)
                print(f"Frame {tval*100:.0f}: u mean={u_mean:.6f}, ")

        return u_frames
