import torch
import torch.nn as nn
import time
import numpy as np

import os
os.sys.path.append("..")

from result_data import result_data
from result_frames import result_frames
# from comparison import compare_trained
from heat_pinn_python.pinn import PINN, PINNConfig
# from pinn_trainer import predict_u_and_derivs
from heat_pinn_python.pinn import load_model, temp_transform

class HeatPINNSolver():
    """
    Solver that uses a Physics-Informed Neural Network (PINN) to approximate
    the solution of the 2D heat equation on a rectangular domain.

    This solver assumes that a model has already been trained and saved to disk.
    It does not perform training; it only loads and evaluates the neural network.
    """
    def __init__(self, dt = 1.0):
        self.dt = dt
        pass

    def pipeline(ibvp, frame, t_steps_per_frame = 1, n_frames = 1):
        """
        Compute time-dependent solution frames using a pretrained PINN model.

        Parameters
        ----------
        ibvp : object
            Problem specification providing:
              - initial_u(x, y)
              - heat_source(x, y, t) (not used here in prediction mode)
              - alpha, a, b, c etc. (not directly used in inference)
        frame : object
            Grid and time settings:
              - nx, ny : number of spatial grid points
              - lx, ly : physical dimensions of the domain
              - nt : number of time steps
              - lt : final time
        t_steps_per_frame : int
            Unused here (included only for API consistency).
        n_frames : int
            Number of time values for which to evaluate and return solution frames.

        Returns
        -------
        u_frames : list of ndarray
            List of predicted temperature fields, each shaped (ny, nx).
        u_means : list of float
            Mean temperature value per predicted frame.
        """
        print("PINN solver")

        
        
        # NN architecture for inference
        hid_layers = 5
        nodes = 50
        model = PINN(hid_layers,nodes).to(device)
        pinn_cfg = PINNConfig()

        # ----------------------------------------
        # Modell (Fourier-PINN oder MLP je nach pinn.py)
        # ----------------------------------------
        model = PINN(
            hid_layers=pinn_cfg.n_hid_layers,
            neurons=pinn_cfg.n_neurons,
            activation=nn.Tanh(),
            use_fourier=pinn_cfg.use_fourier,
            m_fourier=pinn_cfg.m_fourier,
            fourier_scale=pinn_cfg.fourier_scale
        ).to(device)
        model, ckpt = load_model('case3_models/pinn_model2.pt', pinn_cfg, device)
        # model.load_state_dict(torch.load('case3_models/pinn_model2.pt'))
        model.eval()

        print("Model successfully loaded.")

        # Create evaluation grid
        x_vis = torch.linspace(0.0, frame.lx, frame.nx)
        y_vis = torch.linspace(0.0, frame.ly, frame.ny)
        t_vis = torch.linspace(0.0, frame.lt, frame.nt)
        Xv, Yv = torch.meshgrid(x_vis, y_vis, indexing='ij')
        Xv = Xv.flatten()
        Yv = Yv.flatten()

        # Compute initial frame
        u0 = ibvp.initial_u(Xv, Yv).detach().cpu().reshape(frame.ny, frame.nx).numpy()
        f = ibvp.heat_source(Xv, Yv).detach().cpu().reshape(frame.ny, frame.nx).numpy()
        u_frames = [result_data(u0)]

        # Evaluate model at requested times
        dt = 3.0
        with_derivs = True

        if with_derivs:
            for n_frame in range(n_frames):
                start = time.time()
                tval = frame.lt*(1+n_frame)/n_frames  # current time value
                xv = Xv.unsqueeze(1)
                yv = Yv.unsqueeze(1)
                tv = torch.full_like(Xv, tval).unsqueeze(1)
                xyt = torch.cat((xv, yv, tv), dim=1)
                u, u_t_val, lap_u = PINN.predict_u_and_derivs(model, xyt)
                u = temp_transform.inv_scale(u.reshape(frame.ny, frame.nx).detach().numpy())
                u_t = temp_transform.inv_scale(u_t_val.reshape(frame.ny, frame.nx).detach().numpy())
                lap_u = temp_transform.inv_scale(lap_u.reshape(frame.ny, frame.nx).detach().numpy())
                u_frames.append(result_data(u=u, u_t= u_t, lap_u = lap_u))
                min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
                max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
                print(f"Frame {tval:.2f}: mean={u.mean():.6f}, min={u.min():.6f} @ {min_idx}, max={u.max():.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")
            result = result_frames(u_frames, f, has_u_t= True, has_derivs= False, has_laplacian= True)
        else:
            with torch.no_grad():
                for n_frame in range(n_frames):
                    start = time.time()
                    tval = frame.lt*(1+n_frame)/n_frames  # current time value
                    xv = Xv.unsqueeze(1)
                    yv = Yv.unsqueeze(1)
                    tv = torch.full_like(Xv, tval).unsqueeze(1)
                    u = temp_transform.inv_scale(model(xv, yv, tv).reshape(frame.ny, frame.nx).cpu().numpy())
                    u2 = u - temp_transform.inv_scale(model(xv, yv, tv + dt).reshape(frame.ny, frame.nx).cpu().numpy())
                    u_t = (u2 - u) / dt
                    u_frames.append(result_data(u=u, u_t= u_t, lap_u = lap_u))
                    min_idx = tuple(int(i) for i in np.unravel_index(np.argmin(u), u.shape))
                    max_idx = tuple(int(i) for i in np.unravel_index(np.argmax(u), u.shape))
                    print(f"Frame {tval:.2f}: mean={u.mean():.6f}, min={u.min():.6f} @ {min_idx}, max={u.max():.6f} @ {max_idx}, Time needed {time.time() - start:.4f}")
            result = result_frames(u_frames, f, has_u_t= False, has_derivs= False, has_laplacian= False)
        return result


# ------------------------------------------------------------
# Utility functions for training workflows (not used in inference)
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available () else "cpu")

def set_seed(seed=42):
    """
    Set deterministic random seeds for reproducible neural network training.
    """
    torch.manual_seed(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)

def generate_data(n_points, length, total_time, seed=None, device='cpu'):
    """
    Generate random interior and boundary sample points (x, y, t)
    for PINN training.

    Returns
    -------
    x, y, t : tensors of shape (N, 1)
        Points in the spatio-temporal domain used for PDE and boundary residuals.
    """
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
