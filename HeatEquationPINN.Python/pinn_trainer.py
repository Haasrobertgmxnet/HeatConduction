# pinn_trainer.py

import os
os.sys.path.append("..")

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn import PINN
from sampling_service import SamplingService, SamplingConfig
from sampling_service import make_affine_transform

from ibvp_data import ibvp1, IBVPData

from plot_tools import animate_heatmap

from training_utils import (
    apply_warmup,
    apply_scheduler,
    apply_adaptive_lr,
    maybe_clip_gradients,
    log_training,
    early_stopping,
    early_stopping_t,
    load_model,
    loss_number
)

# =============================================================================
# KONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    # Loss-Gewichte
    lambda_phy: float = 1.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_cont: float = 1.0

    # Learning Rate and Schedules
    lr: float = 1e-3 # 5e-4
    min_lr: float = 1e-6
    max_lr: float = 5e-3
    warmup_steps: int = 500

    use_warmup: bool = False
    use_cosine_annealing: bool = True
    use_adaptive_lr: bool = True

    # Gradient Clipping
    use_gradient_clipping: bool = False
    grad_clip_norm: float = 1.0

    # Logging
    use_logging: bool = True
    log_interval: int = 100

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 2000
    min_delta: float = 1e-6
    ema_alpha: float = 0.1

    # Epochen
    epochs: int = 12000


@dataclass
class PINNConfig:
    n_hid_layers: int = 5
    n_neurons: int = 50
    use_fourier = False
    m_fourier: int = 12
    fourier_scale: float = 2.0


# =============================================================================
# Autograd-Helfer
# =============================================================================

@torch.no_grad()
def predict_u(model, xyt: torch.Tensor) -> torch.Tensor:
    """
    Schnelle Vorhersage nur von u(x,y,t).
    Kein Autograd notwendig, ideal für Animationen und Auswertung.
    """
    return model(xyt)[:, 0:1]


def predict_u_and_derivs(model, xyt: torch.Tensor):
    if not xyt.requires_grad:
        xyt = xyt.requires_grad_(True)

    u = model(xyt)

    grads = torch.autograd.grad(
        u,
        xyt,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(
        u_x,
        xyt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y,
        xyt,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]

    lap_u = u_xx + u_yy
    return u, u_t, lap_u



# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def training_pipeline():

    print("Training gestartet...")
    seed = 35
    torch.manual_seed(seed)

    training_cfg = TrainingConfig()
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

    optimizer = optim.Adam(model.parameters(), lr=training_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_cfg.epochs - training_cfg.warmup_steps),
        eta_min=training_cfg.min_lr
    )

    use_trained_model = True
    start_epoch = 0
    if os.path.exists("pinn_model.pt") and use_trained_model:
        model, ckpt = load_model("pinn_model.pt", pinn_cfg, device)

        optimizer = optim.Adam(model.parameters(), lr=training_cfg.lr)
        optimizer.load_state_dict(ckpt["optimizer_state"])

        start_epoch = ckpt["epoch"] + 1

    # ----------------------------------------
    # Loss-Funktionen
    # ----------------------------------------

    def heat_source(xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        return heat_problem.heat_source(x, y, t)

    def initial_u(xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        return heat_problem.initial_u(x, y)

    def physics_loss(model: PINN, xyt: torch.Tensor) -> torch.Tensor:
        _, u_t_val, lap_u = predict_u_and_derivs(model, xyt)
        f = heat_source(xyt)
        return ((u_t_val - alpha * lap_u - f) ** 2).mean()

    def ic_loss(model: PINN, Xyt_ic: torch.Tensor) -> torch.Tensor:
        u_pred, _, _ = predict_u_and_derivs(model, Xyt_ic)
        u_ic = initial_u(Xyt_ic)
        return ((u_pred - u_ic) ** 2).mean()

    def boundary_loss(model: PINN, xyt_bnd: torch.Tensor) -> torch.Tensor:
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
        return ((a * u_b + scal * b * du_dn - c) ** 2).mean()

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
    B = np.diag([1.0, 1.0, 45.0])
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

    for epoch in range(start_epoch, start_epoch + training_cfg.epochs):

        time_pt1 = time.time()
        if epoch % 100 == 0 and epoch > 0:
            sampler.resample_pools()

        xyt_int = sampler.sample_interior()
        xyt_ic = sampler.sample_initial()
        xyt_ic_eps = xyt_ic.clone()
        xyt_ic_eps[:, 2] = 1e-3  # z.B. 1e-3

        xyt_bnd, normals = sampler.sample_boundary()

        time_pt2 = time.time()
        time_samp = time_pt2 - time_pt1

        model.train()
        apply_warmup(optimizer, epoch, training_cfg)

        # Verluste berechnen
        loss_phy = physics_loss(model, xyt_int)
        loss_ic = ic_loss(model, xyt_ic)
        loss_cont = ic_loss(model, xyt_ic_eps)
        loss_bc = boundary_loss(model, xyt_bnd)

        losses_phy.append(loss_phy.item())
        losses_ic.append(loss_ic.item())
        losses_bc.append(loss_bc.item())
        losses_cont.append(loss_cont.item())

        ln_phy = loss_number(losses_phy)
        ln_ic = loss_number(losses_ic)
        ln_bc = loss_number(losses_bc)
        ln_cont = loss_number(losses_cont)

        sum_ln = ln_phy + ln_ic + ln_bc + ln_cont

        scal = 4.0
        training_cfg.lambda_phy = scal*ln_phy / sum_ln
        training_cfg.lambda_ic = scal*ln_ic / sum_ln
        training_cfg.lambda_bc = scal*ln_bc / sum_ln
        training_cfg.lambda_cont = scal*ln_cont / sum_ln

        losses_float = (
            loss_phy.item(),
            loss_ic.item(),
            loss_bc.item(),
            loss_cont.item()
        )

        # Gesamtverlust
        total_loss = (
            training_cfg.lambda_phy * loss_phy +
            training_cfg.lambda_ic * loss_ic +
            training_cfg.lambda_bc * loss_bc +
            training_cfg.lambda_cont * loss_cont
        )

        losses.append(total_loss)

        optimizer.zero_grad()
        total_loss.backward()

        maybe_clip_gradients(model, training_cfg)
        optimizer.step()

        apply_scheduler(scheduler, epoch, training_cfg)
        apply_adaptive_lr(optimizer, loss_phy.item(), state, training_cfg)

        time_train = time.time() - time_pt2

        # --- erweiterte Metriken (für Logging) ---
        model.eval()
        u_vals, u_t_vals, lap_vals = predict_u_and_derivs(model, xyt_int)
        residual = (u_t_vals - alpha * lap_vals - heat_source(xyt_int))

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
            "lambda_phy": training_cfg.lambda_phy,
            "lambda_ic": training_cfg.lambda_ic,
            "lambda_bc": training_cfg.lambda_bc,
            "time_samp": time_samp,
            "time_train": time_train
        }
        temperatures.append(metrics["u_max"])
        best_metrics = metrics

        log_training(epoch, losses_float, metrics, training_cfg)

        # Early Stopping
        if early_stopping(losses, training_cfg):
           break
        if early_stopping_t(temperatures):
           break

    #
    # ----------------------------------------
    # Modell speichern
    # ----------------------------------------
    save_path = "pinn_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "training_cfg": training_cfg
    }, save_path)

    print(f"Modell gespeichert unter {save_path}")

    # ----------------------------------------
    # Training Report
    # ----------------------------------------

    duration = time.time() - start_time
    print("\n===== TRAINING REPORT =====")
    best_key = "best_loss" if "best_loss" in state else "best"
    best_val = state.get(best_key, float("nan"))
    print(f"Beste Loss (total, EMA/best): {best_val:.3e}")
    print(f"Final λ_phy: {training_cfg.lambda_phy:.3f}")
    if best_metrics is not None:
        print(f"Final PDE-Residual mean: {best_metrics['res_mean']:.3e}")
        print(f"Final u in [{best_metrics['u_min']:.3f}, {best_metrics['u_max']:.3f}]")
    print(f"Trainingsdauer: {duration:.2f} s")
    print("===========================\n")

    # ----------------------------------------
    # Optionale Visualisierung / Animation
    # ----------------------------------------
    t_vals = torch.linspace(0.0, 60.0, 100)
    animate_heatmap(model, t_vals, grid_size=80, device=device)
