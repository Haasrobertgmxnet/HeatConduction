import os
os.sys.path.append("..")

from solver_python.ibvp_data import ibvp1, IBVPData
from solver_python.plot_tools import animate_heatmap

# from sqlite3 import SQLITE_CANTOPEN_DIRTYWAL

import json
import time
from dataclasses import dataclass
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn import PINN
from pinn import PINNConfig
from pinn import load_model, temp_transform

from sampling_service import SamplingService, SamplingConfig
from sampling_service import make_affine_transform

from training_phase_config import TrainingPhaseConfig
from training_phase_config import training_phase1
from training_phase_config import training_phase2

from training_utils import (
    apply_warmup,
    apply_scheduler,
    apply_adaptive_lr,
    maybe_clip_gradients,
    log_training,
    early_stopping,
    early_stopping_t,
    update_lambda_phy
)


# =============================================================================
# KONFIGURATION
# =============================================================================
@dataclass
class __TrainingConfig:

    # -------------------------------
    # Loss-Gewichte (Startwerte)
    # -------------------------------
    lambda_phy: float = 0.05     # WICHTIG: stark reduziert
    lambda_ic: float  = 1.0
    lambda_bc: float  = 1.0
    lambda_cont: float = 1.0

    # -------------------------------
    # Learning Rate
    # -------------------------------
    lr: float = 3e-3             # höherer Start
    min_lr: float = 5e-5
    max_lr: float = 3e-3
    warmup_steps: int = 800

    use_warmup: bool = True
    use_cosine_annealing: bool = True
    use_adaptive_lr: bool = False   # ❗ ausschalten

    # -------------------------------
    # Gradient Handling
    # -------------------------------
    use_gradient_clipping: bool = True
    grad_clip_norm: float = 0.5

    # -------------------------------
    # Logging
    # -------------------------------
    use_logging: bool = True
    log_interval: int = 100

    # -------------------------------
    # Early stopping
    # -------------------------------
    use_early_stopping: bool = False  # ❗ sonst stoppt es zu früh
    patience: int = 2000
    min_delta: float = 1e-6
    ema_alpha: float = 0.05

    # -------------------------------
    # Training
    # -------------------------------
    epochs: int = 8000 # 15000

# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def training_pipeline():

    print("Training gestartet...")
    seed = 35
    torch.manual_seed(seed)

    pinn_cfg = PINNConfig()

    # IBVP-Daten kopieren
    heat_problem: IBVPData = ibvp1.copy()
    alpha = heat_problem.alpha

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    optimizer = optim.Adam(model.parameters(), lr=training_phase1.lr)

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_phase1.epochs - training_phase1.warmup_steps),
        eta_min=training_phase1.min_lr
    )

    scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.995
    )

    update_lambda1 = update_lambda_phy
    update_lambda2 = None

    def heat_source(xyt: torch.Tensor) -> torch.Tensor:
        global temp_transform
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        return temp_transform.scale_heat_source(f = heat_problem.heat_source(x, y, t))

    def initial_u(xyt: torch.Tensor) -> torch.Tensor:
        global temp_transform
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        return temp_transform.scale(heat_problem.initial_u(x, y))

    # ----------------------------------------
    # Loss-Funktionen
    # ----------------------------------------
    def physics_loss(model: PINN, xyt: torch.Tensor) -> torch.Tensor:
        _, u_t_val, lap_u = PINN.predict_u_and_derivs(model, xyt)
        f = heat_source(xyt)
        return ((u_t_val - alpha * lap_u - f) ** 2).mean()

    def ic_loss(model: PINN, Xyt_ic: torch.Tensor) -> torch.Tensor:
        u_pred, _, _ = PINN.predict_u_and_derivs(model, Xyt_ic)
        u_ic = initial_u(Xyt_ic)
        return ((u_pred - u_ic) ** 2).mean()

    def boundary_loss(model: PINN, xyt_bnd: torch.Tensor, normals = None) -> torch.Tensor:
        global temp_transform
        xyt_req = xyt_bnd.clone().detach().requires_grad_(True)
        u_b = model(xyt_req)

        grads = torch.autograd.grad(
            outputs=u_b,
            inputs=xyt_req,
            grad_outputs=torch.ones_like(u_b),
            create_graph=True
        )[0]

        du_dn = torch.sum(grads * normals, dim=1, keepdim=True)
        a, b, c = heat_problem.a, heat_problem.b, heat_problem.c

        # du/dn leicht skaliert, um Extremwerte abzufangen
        scal = 1.0
        return ((a * u_b + scal * b * du_dn - temp_transform.scale_bc_c(a, c)) ** 2).mean()

    # ----------------------------------------
    # Training
    # ----------------------------------------

    state = {}
    best_metrics = None
    start_time = time.time()
    losses = []
    temperatures = []

    losses_phy = []
    losses_ic = []
    losses_bc = []
    losses_cont = []

    # ----------------------------------------
    # Sampling
    # ----------------------------------------
    a = np.zeros(3)
    B = np.diag([1.0, 1.0, 65.0])
    tr = make_affine_transform(a, B)

    sampler = SamplingService(
        SamplingConfig(
            n_interior_pool=200000,
            n_boundary_pool=20000,
            frac_interior=0.02,
            frac_boundary=0.02,
        ),
        transform= tr,
        mode="biased"
    )

    # ----------------------------------------
    # Modell speichern
    # ----------------------------------------
    def save_model(model : nn.Module, training_phase_cfg : TrainingPhaseConfig):
        save_path = training_phase_cfg.model_file
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": training_phase_cfg.epochs
        }, save_path)

        print(f"Modell gespeichert unter {save_path}")

    # ----------------------------------------
    # Trainingsschleife
    def train_loop(training_phase_cfg : TrainingPhaseConfig, optimizer, scheduler = None, update_lambda = None):
        if training_phase_cfg.use_trained_model and os.path.exists(training_phase_cfg.model_file):
            nonlocal model
            model, ckpt = load_model(training_phase_cfg.model_file, pinn_cfg, device)
            print(f"Vorgebildetes Modell aus {training_phase_cfg.model_file} geladen.")
            return
        print("Starte Trainingsphase mit Konfiguration:")
        for g in optimizer.param_groups:
            g["lr"] = training_phase_cfg.lr
        print("New config")
        print(f"Learning Rate: {training_phase_cfg.lr}")
        print(asdict(training_phase_cfg))

        for epoch in range(0, training_phase_cfg.epochs):
            time_pt1 = time.time()
            if epoch % 100 == 0 and epoch > 0:
                sampler.resample_pools()

            xyt_int = sampler.sample_interior()
            xyt_ic = sampler.sample_initial()
            xyt_ic_eps = xyt_ic.clone()
            xyt_ic_eps[:, 2] = 1e-3

            xyt_bnd, normals = sampler.sample_boundary()

            time_pt2 = time.time()
            time_samp = time_pt2 - time_pt1

            model.train()
            apply_warmup(optimizer, epoch, training_phase_cfg)

            # Verluste berechnen
            loss_phy = physics_loss(model, xyt_int)
            loss_ic = ic_loss(model, xyt_ic)
            loss_cont = ic_loss(model, xyt_ic_eps)
            loss_bc = boundary_loss(model, xyt_bnd, normals)

            losses_phy.append(loss_phy.item())
            losses_ic.append(loss_ic.item())
            losses_bc.append(loss_bc.item())
            losses_cont.append(loss_cont.item())

            if update_lambda  is not None:
                training_phase_cfg.lambda_phy = update_lambda(epoch)

            losses_float = (
                loss_phy.item(),
                loss_ic.item(),
                loss_bc.item(),
                loss_cont.item()
            )

            # Gesamtverlust
            total_loss = (
                training_phase_cfg.lambda_phy * loss_phy +
                training_phase_cfg.lambda_ic * loss_ic +
                training_phase_cfg.lambda_bc * loss_bc +
                training_phase_cfg.lambda_cont * loss_cont
            )

            losses.append(total_loss)

            optimizer.zero_grad()
            total_loss.backward()

            maybe_clip_gradients(model, training_phase_cfg)
            optimizer.step()

            if scheduler is not None:
                apply_scheduler(scheduler, epoch, training_phase_cfg)

            apply_adaptive_lr(optimizer, loss_phy.item(), state, training_phase_cfg)

            time_train = time.time() - time_pt2

            # --- erweiterte Metriken (für Logging) ---
            model.eval()
            u_vals, u_t_vals, lap_vals = PINN.predict_u_and_derivs(model, xyt_int)
            residual = (u_t_vals - alpha * lap_vals - heat_source(xyt_int))

            u_vals = temp_transform.inv_scale(u_vals)
            u_t_vals = temp_transform.inv_scale(u_t_vals)
            lap_vals = temp_transform.inv_scale(lap_vals)

            metrics = {
                "u_min": u_vals.min().item(),
                "u_max": u_vals.max().item(),
                "u_t_min": u_t_vals.min().item(),
                "u_t_max": u_t_vals.max().item(),
                "lap_min": lap_vals.min().item(),
                "lap_max": lap_vals.max().item(),
                "res_min": residual.min().item(),
                "res_max": residual.max().item(),
                "res_mean": residual.abs().mean().item(),
                "lambda_phy": training_phase_cfg.lambda_phy,
                "lambda_ic": training_phase_cfg.lambda_ic,
                "lambda_bc": training_phase_cfg.lambda_bc,
                "time_samp": time_samp,
                "time_train": time_train
            }
            temperatures.append(metrics["u_max"])
            best_metrics = metrics

            log_training(epoch, losses_float, metrics, training_phase_cfg)

            # Early Stopping
            if training_phase_cfg.use_early_stopping and early_stopping(losses, training_phase_cfg):
               break

        print("Trainingphase beendet.")
        save_model(model = model, training_phase_cfg = training_phase_cfg) 
        return best_metrics

    best_metrics = train_loop(training_phase_cfg = training_phase1, optimizer = optimizer, scheduler = scheduler1, update_lambda = update_lambda_phy)
    best_metrics = train_loop(training_phase_cfg = training_phase2, optimizer = optimizer, scheduler = scheduler2)

    # ----------------------------------------
    # Training Report
    # ----------------------------------------
    duration = time.time() - start_time
    print("\n===== TRAINING REPORT =====")
    best_key = "best_loss" if "best_loss" in state else "best"
    best_val = state.get(best_key, float("nan"))
    print(f"Beste Loss (total, EMA/best): {best_val:.3e}")
    # print(f"Final λ_phy: {training_cfg.lambda_phy:.3f}")
    if best_metrics is not None:
        print(f"Final PDE-Residual mean: {best_metrics['res_mean']:.3e}")
        print(f"Final u in [{best_metrics['u_min']:.3f}, {best_metrics['u_max']:.3f}]")
    print(f"Trainingsdauer: {duration:.2f} s")
    print("===========================\n")

    # ----------------------------------------
    # Optionale Visualisierung / Animation
    # ----------------------------------------
    t_vals = torch.linspace(0.0, 60.0, 100)
    animate_heatmap(model, t_vals, grid_size=80, device=device, temp_transform= temp_transform)
    
