from dataclasses import dataclass
import os

@dataclass
class TrainingPhaseConfig:
    json_file: str = "training_phase_cfg.json"
    model_file: str ="pinn_model.pt"
    use_trained_model: bool = True

    # -------------------------------
    # Loss-Gewichte (Startwerte)
    # -------------------------------
    lambda_phy: float = 0.05
    lambda_ic: float  = 1.0
    lambda_bc: float  = 1.0
    lambda_cont: float = 1.0

    # -------------------------------
    # Learning Rate
    # -------------------------------
    lr: float = 3e-3
    min_lr: float = 5e-5
    max_lr: float = 3e-3
    warmup_steps: int = 800

    use_warmup: bool = True
    use_cosine_annealing: bool = True
    use_adaptive_lr: bool = False

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
    use_early_stopping: bool = False
    patience: int = 2000
    min_delta: float = 1e-6
    ema_alpha: float = 0.05

    # -------------------------------
    # Training
    # -------------------------------
    epochs: int = 8000 # 15000

training_phase1 = TrainingPhaseConfig(
    json_file = "training_phase1_cfg.json",
    model_file = "pinn_model1.pt",
    use_trained_model = True,

    # Loss-Gewichte
    lambda_phy=0.05,
    lambda_ic=1.0,
    lambda_bc=1.0,
    lambda_cont=1.0,

    # Learning Rate
    lr=3e-3,
    min_lr=5e-5,
    max_lr=3e-3,
    warmup_steps=800,

    use_warmup=True,
    use_cosine_annealing=True,
    use_adaptive_lr=False,

    # Gradient Handling
    use_gradient_clipping=True,
    grad_clip_norm=0.5,

    # Logging
    use_logging=True,
    log_interval=100,

    # Early stopping
    use_early_stopping=False,
    patience=2000,
    min_delta=1e-6,
    ema_alpha=0.05,

    # Training
    epochs=8000
)

training_phase2 = TrainingPhaseConfig(
    json_file = "training_phase2_cfg.json",
    model_file ="pinn_model2.pt",
    use_trained_model = False,

    # Loss-Gewichte
    lambda_phy = 0.8,
    lambda_bc  = 0.5,
    lambda_ic  = 0.3,
    lambda_cont= 0.5,

    # Learning Rate
    lr = 1e-5,
    min_lr = 5e-6,

    use_cosine_annealing = False,
    use_adaptive_lr = False,
    max_lr=3e-3,
    warmup_steps=800,
    use_warmup=False,

    # Gradient Handling
    use_gradient_clipping=False,
    grad_clip_norm=0.5,

    # Logging
    use_logging=True,
    log_interval=100,

    # Early stopping
    use_early_stopping=False,
    patience=2000,
    min_delta=1e-6,
    ema_alpha=0.05,

    # Training
    epochs=3000
)
