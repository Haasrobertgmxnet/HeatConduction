# training_utils.py
import torch
import numpy as np

import torch
import torch.nn as nn
from pinn import PINN

# ======================================================================
#  WARMUP (linear)
# ======================================================================

def apply_warmup(optimizer, epoch, cfg):
    """
    Linear Warmup über cfg.warmup_steps.
    """
    if not cfg.use_warmup:
        return

    if epoch < cfg.warmup_steps:
        lr = cfg.lr * float(epoch + 1) / float(cfg.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr


# ======================================================================
#  COSINE ANNEALING
# ======================================================================

def apply_scheduler(scheduler, epoch, cfg):
    """
    Cosine-Annealing nach Warmup.
    """
    if not cfg.use_cosine_annealing:
        return

    if epoch >= cfg.warmup_steps:
        scheduler.step()


# ======================================================================
#  ADAPTIVE PHYSICS-BASED LR
# ======================================================================

def apply_adaptive_lr(optimizer, L_phy_val, state, cfg):
    """
    PINN-spezifische LR-Adaption.
    """
    if not cfg.use_adaptive_lr:
        return

    last = state.get("L_phy_last")
    if last is None:
        state["L_phy_last"] = L_phy_val
        return

    ratio = L_phy_val / (last + 1e-12)

    for g in optimizer.param_groups:
        lr = g["lr"]

        if ratio > 1.3:            # schlechter geworden → LR senken
            lr = max(lr * 0.5, cfg.min_lr)

        elif ratio < 0.7:          # stark verbessert → LR leicht erhöhen
            lr = min(lr * 1.05, cfg.max_lr)

        g["lr"] = lr

    state["L_phy_last"] = L_phy_val


# ======================================================================
#  GRADIENT CLIPPING
# ======================================================================

def maybe_clip_gradients(model, cfg):
    """
    Verhindert Runaway-Gradients.
    """
    if cfg.use_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)


# ======================================================================
#  LOGGING
# ======================================================================

def log_training(epoch, losses, metrics, cfg):
    """
    losses = (L_phy, L_ic, L_bc, L_cont)
    metrics enthält min/max der Modell-Ableitungen.
    """
    if not cfg.use_logging:
        return

    if epoch % cfg.log_interval != 0:
        return

    L_phy, L_ic, L_bc, L_cont = losses
    m = metrics

    print(
        f"[Epoch {epoch:5d}] | "
        f"PHY={L_phy:.3e}, IC={L_ic:.3e}, BC={L_bc:.3e}, CONT={L_cont:.3e} | "
        f"u=[{m['u_min']:.3f}, {m['u_max']:.3f}] | "
        f"u_t=[{m['u_t_min']:.3f}, {m['u_t_max']:.3f}] | "
        f"lap=[{m['lap_min']:.3f}, {m['lap_max']:.3f}] | "
        f"l_phy={m['lambda_phy']:.5e} | "
        f"l_ic={m['lambda_ic']:.5e} | "
        f"l_bc={m['lambda_bc']:.5e}  | "
        f"sampling time = {m['time_samp']:.3f}  | "
        f"training time = {m['time_train']:.3f}"
    )


# ======================================================================
#  EARLY STOPPING (stabil, EMA-basiert)
# ======================================================================

def early_stopping(losses, cfg):
    if len(losses)<2:
        return False
    current_loss = losses[-1]
    if losses[-1] < losses[-2] - 0*1e-20:  # 1e-6 = kleine Toleranz
            losses = []
            losses.append(current_loss)
            # trigger_times = 0
            # Optional: bestes Modell speichern
            # torch.save(model.state_dict(), "best_model.pt")
            return False
    else:
        if len(losses) >= cfg.patience:
            print(f"Early stopping triggered (patience={cfg.patience}). ")
            return True
    return False

def early_stopping_t(temperatures):
    if len(temperatures)<5001:
        return False
    current_temperature = temperatures[-1]
    if current_temperature - temperatures[-2] > 3.0:  # 1e-6 = kleine Toleranz
            temperatures = []
            temperatures.append(current_temperature)
            # trigger_times = 0
            # Optional: bestes Modell speichern
            # torch.save(model.state_dict(), "best_model.pt")
            return False
    else:
        if len(temperatures) >= 200:
            print(f"Early stopping triggered: Temperatures.")
            return True
    return False

def __early_stopping(total_loss, state, cfg):
    """
    Robustes PINN-Early-Stopping mit:
      - EMA-Glättung
      - min_delta
      - Geduld (patience)

    Gibt True zurück, wenn Training abbrechen soll.
    """
    if not cfg.use_early_stopping:
        return False

    # Initialwerte
    ema = state.setdefault("ema", float("inf"))
    best = state.setdefault("best", float("inf"))
    wait = state.setdefault("wait", 0)

    # EMA smoothing
    alpha = getattr(cfg, "ema_alpha", 0.1)
    ema = alpha * total_loss + (1 - alpha) * ema
    state["ema"] = ema

    min_delta = getattr(cfg, "min_delta", 1e-6)

    if ema < best - min_delta:
        # Verbesserung erkannt
        state["best"] = ema
        state["wait"] = 0
    else:
        # keine Verbesserung
        state["wait"] += 1

    # Geduld überschritten?
    if state["wait"] >= cfg.patience:
        print(
            f"Early stopping triggered (patience={cfg.patience}). "
            f"Best EMA loss={best:.3e}"
        )
        return True

    return False


# ======================================================================
#  DYNAMISCHE λ-ANPASSUNG
# ======================================================================

def update_lambda_phy(epoch):
    if epoch < 500:
        return 0.05
    elif epoch < 1500:
        return 0.2
    elif epoch < 3000:
        return 0.5
    else:
        return 1.0

def _loss_number(losses):
    # return 1.0
    wnd_width = 10
    if len(losses)< wnd_width:
        return 1.0
    losses_window = losses[-wnd_width:-1]
    l_mean = np.mean(losses_window)
    return l_mean
    l_max = np.max(losses_window)
    l_diff = losses_window[-1] - losses_window[0]
    ex = 2
    return ((l_max - l_diff)/l_max)**ex

