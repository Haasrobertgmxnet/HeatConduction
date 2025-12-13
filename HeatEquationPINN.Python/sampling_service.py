# sampling_service.py

import torch
from dataclasses import dataclass
from typing import Optional, Callable


# --------------------------------------------------
# Config
# --------------------------------------------------
@dataclass
class SamplingConfig:
    n_interior_samples: int = 3000
    n_initial_samples: int = 100
    n_boundary_samples: int = 100


# --------------------------------------------------
# Optional affine transform (torch-compatible)
# v ↦ a + v @ Bᵀ
# --------------------------------------------------
def make_affine_transform(a, B):
    a = torch.tensor(a, dtype=torch.float32).view(1, 3)
    B = torch.tensor(B, dtype=torch.float32).view(3, 3)

    def transform(v: torch.Tensor):
        return a.to(v.device) + v @ B.T.to(v.device)

    return transform


# --------------------------------------------------
# SamplingService
# --------------------------------------------------
class SamplingService:
    """
    Torch-first SamplingService

    mode:
        "uniform" : uniform interior sampling
        "biased"  : oversample t <= t_eps
    """

    def __init__(
        self,
        config: SamplingConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "uniform",
        frac_near0: float = 0.3,
        t_eps: float = 0.02,
        dtype=torch.float32,
    ):
        self.cfg = config
        self.transform = transform
        self.mode = mode
        self.frac_near0 = frac_near0
        self.t_eps = t_eps
        self.dtype = dtype

    # --------------------------------------------------
    # Core random helper
    # --------------------------------------------------
    def _rand_wrap(self, n: int, dims: int, device):
        r1 = torch.rand((n, dims), device=device, dtype=self.dtype)
        r2 = torch.rand((n, dims), device=device, dtype=self.dtype)
        return torch.where(r1 > r2, 0.5 * (1.0 + r1), 0.5 * (1.0 - r2))

    # --------------------------------------------------
    # Interior sampling
    # --------------------------------------------------
    def _sample_interior_uniform(self, device):
        xyt = self._rand_wrap(self.cfg.n_interior_samples, 3, device)
        return self.transform(xyt) if self.transform else xyt

    def _sample_interior_biased(self, device):
        n = self.cfg.n_interior_samples

        # x,y
        xy = self._rand_wrap(n, 2, device)

        # mixture for t
        u = torch.rand((n, 1), device=device)
        v = torch.rand((n, 1), device=device)

        t_near = v * self.t_eps
        t_far = self.t_eps + v * (1.0 - self.t_eps)
        t = torch.where(u < self.frac_near0, t_near, t_far)

        xyt = torch.cat([xy, t], dim=1)
        return self.transform(xyt) if self.transform else xyt

    def sample_interior(self, device):
        if self.mode == "uniform":
            return self._sample_interior_uniform(device)
        elif self.mode == "biased":
            return self._sample_interior_biased(device)
        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")

    # --------------------------------------------------
    # Initial condition (t = 0)
    # --------------------------------------------------
    def sample_initial(self, device):
        xy = self._rand_wrap(self.cfg.n_initial_samples, 2, device)
        t = torch.zeros((self.cfg.n_initial_samples, 1), device=device)
        xyt = torch.cat([xy, t], dim=1)
        return self.transform(xyt) if self.transform else xyt

    # --------------------------------------------------
    # Boundary sampling
    # --------------------------------------------------
    def sample_boundary(self, device):
        n = self.cfg.n_boundary_samples

        yt = self._rand_wrap(n, 2, device)
        xt = self._rand_wrap(n, 2, device)

        left = torch.cat([torch.zeros((n, 1), device=device), yt], dim=1)
        right = torch.cat([torch.ones((n, 1), device=device), yt], dim=1)

        bottom = torch.cat(
            [xt[:, 0:1], torch.zeros((n, 1), device=device), xt[:, 1:2]], dim=1
        )
        top = torch.cat(
            [xt[:, 0:1], torch.ones((n, 1), device=device), xt[:, 1:2]], dim=1
        )

        xyt = torch.cat([left, right, bottom, top], dim=0)
        return self.transform(xyt) if self.transform else xyt

    # --------------------------------------------------
    # Boundary normals
    # --------------------------------------------------
    def sample_normals(self, device):
        n = self.cfg.n_boundary_samples

        left = torch.tensor([-1, 0, 0], device=device).repeat(n, 1)
        right = torch.tensor([1, 0, 0], device=device).repeat(n, 1)
        bottom = torch.tensor([0, -1, 0], device=device).repeat(n, 1)
        top = torch.tensor([0, 1, 0], device=device).repeat(n, 1)

        return torch.cat([left, right, bottom, top], dim=0)

    # --------------------------------------------------
    # Main entry used by training
    # --------------------------------------------------
    def get_samples(self, heat_problem):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Interior
        xyt_interior = self.sample_interior(device)

        # Initial condition
        Xyt_ic = self.sample_initial(device)
        x_ic = Xyt_ic[:, 0:1]
        y_ic = Xyt_ic[:, 1:2]

        u_ic = heat_problem.initial_u(x_ic, y_ic)
        if u_ic.dim() == 1:
            u_ic = u_ic.unsqueeze(1)

        # Continuity points
        eps_time = 1e-3
        Xyt_ic_eps = Xyt_ic.clone()
        Xyt_ic_eps[:, 2] = eps_time

        # Boundary
        xyt_boundary = self.sample_boundary(device)
        normals = self.sample_normals(device)

        return (
            xyt_interior,
            Xyt_ic,
            xyt_boundary,
            normals,
            u_ic.to(device),
            Xyt_ic_eps,
        )
