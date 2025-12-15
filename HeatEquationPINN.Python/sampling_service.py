# sampling_service.py

import torch
from dataclasses import dataclass
from typing import Optional, Callable


# --------------------------------------------------
# Config
# --------------------------------------------------
@dataclass
class SamplingConfig:
    # Pool sizes
    n_interior_pool: int = 8000
    n_initial_pool: int = 400
    n_boundary_pool: int = 400   # per side

    # Batch fractions
    frac_interior: float = 0.2
    frac_boundary: float = 0.2
    frac_initial: float = 0.2

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
# SamplingService (Pool + Batch)
# --------------------------------------------------
class SamplingService:
    def __init__(
        self,
        cfg: SamplingConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "uniform",     # "uniform" | "biased"
        frac_near0: float = 0.3,
        t_eps: float = 0.02,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.transform = transform
        self.mode = mode
        self.frac_near0 = frac_near0
        self.t_eps = t_eps
        self.dtype = dtype
        self.device = torch.device(device)

        self._build_pools()

    # --------------------------------------------------
    # Random helper
    # --------------------------------------------------
    def _rand_wrap(self, n, dims):
        r1 = torch.rand((n, dims), device=self.device, dtype=self.dtype)
        r2 = torch.rand((n, dims), device=self.device, dtype=self.dtype)
        return torch.where(r1 > r2, 0.5 * (1 + r1), 0.5 * (1 - r2))

    # --------------------------------------------------
    # Pool builders
    # --------------------------------------------------
    def _build_interior_pool(self):
        n = self.cfg.n_interior_pool

        if self.mode == "uniform":
            xyt = self._rand_wrap(n, 3)

        elif self.mode == "biased":
            xy = self._rand_wrap(n, 2)
            u = torch.rand((n, 1), device=self.device)
            v = torch.rand((n, 1), device=self.device)

            t_near = v * self.t_eps
            t_far = self.t_eps + v * (1 - self.t_eps)
            t = torch.where(u < self.frac_near0, t_near, t_far)

            xyt = torch.cat([xy, t], dim=1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.transform:
            xyt = self.transform(xyt)

        self.interior_pool = xyt

    def _build_initial_pool(self):
        n = self.cfg.n_initial_pool
        xy = self._rand_wrap(n, 2)
        t = torch.zeros((n, 1), device=self.device)
        xyt = torch.cat([xy, t], dim=1)

        if self.transform:
            xyt = self.transform(xyt)

        self.initial_pool = xyt

    def _build_boundary_pool(self):
        n = self.cfg.n_boundary_pool

        yt = self._rand_wrap(n, 2)
        xt = self._rand_wrap(n, 2)

        left   = torch.cat([torch.zeros((n,1), device=self.device), yt], dim=1)
        right  = torch.cat([torch.ones((n,1),  device=self.device), yt], dim=1)
        bottom = torch.cat([xt[:,0:1], torch.zeros((n,1), device=self.device), xt[:,1:2]], dim=1)
        top    = torch.cat([xt[:,0:1], torch.ones((n,1),  device=self.device), xt[:,1:2]], dim=1)

        xyt = torch.cat([left, right, bottom, top], dim=0)

        if self.transform:
            xyt = self.transform(xyt)

        self.boundary_pool = xyt

        # normals (fix, no RNG)
        self.boundary_normals = torch.cat([
            torch.tensor([-1,0,0]).repeat(n,1),
            torch.tensor([+1,0,0]).repeat(n,1),
            torch.tensor([0,-1,0]).repeat(n,1),
            torch.tensor([0,+1,0]).repeat(n,1),
        ], dim=0).to(self.device, self.dtype)

    def _build_pools(self):
        self._build_interior_pool()
        self._build_initial_pool()
        self._build_boundary_pool()

    # --------------------------------------------------
    # Batch samplers (HOT PATH)
    # --------------------------------------------------
    def _sample_batch(self, pool, frac):
        n = pool.shape[0]
        k = int(frac * n)
        idx = torch.randperm(n, device=self.device)[:k]
        return pool[idx]

    def sample_interior(self):
        return self._sample_batch(self.interior_pool, self.cfg.frac_interior)

    def sample_initial(self):
        return self._sample_batch(self.initial_pool, self.cfg.frac_initial)

    def sample_boundary(self):
        idx = torch.randperm(self.boundary_pool.shape[0], device=self.device)[
            : int(self.cfg.frac_boundary * self.boundary_pool.shape[0])
        ]
        return self.boundary_pool[idx], self.boundary_normals[idx]

    # --------------------------------------------------
    # Optional: epoch-wise resampling
    # --------------------------------------------------
    def resample_pools(self):
        self._build_pools()
