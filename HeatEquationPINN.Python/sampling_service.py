# --------------------- sampling_service.py (FIXED) ------------------------

from enum import Enum
import numpy as np
import torch

def _rand_wrap():
    r1 = np.random.rand()
    r2 = np.random.rand()
    return 0.5*(1+r1) if r1>r2 else 0.5*(1-r2)

def rand_wrap(dims: int, samples: int):
    return [[_rand_wrap() for _ in range(dims)] for _ in range(samples)]

class SamplingService:
    def __init__(self, transform=None):
        self.transform = transform

    # --------------------------------------------------------
    # internal helper: convert to torch tensor
    # --------------------------------------------------------
    def as_torch(self, data, device=None, dtype=torch.float32):
        """Convert list/array to torch tensor of shape (N,d)."""
        t = torch.tensor(data, dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t

    # --------------------------------------------------------
    # interior points
    # --------------------------------------------------------
    def calc_interior_samples(self, samples: int):
        rnds = rand_wrap(3, samples)
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.interior_samples = rnds

    def interior_torch(self, device=None):
        return self.as_torch(self.interior_samples, device=device)

    # --------------------------------------------------------
    # initial condition points
    # --------------------------------------------------------
    def calc_initial_samples(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[x, y, 0.0] for x, y in rnds]
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.initial_samples = rnds

    def initial_torch(self, device=None):
        return self.as_torch(self.initial_samples, device=device)

    # --------------------------------------------------------
    # boundary samples
    # --------------------------------------------------------
    def calc_boundary_samples_left(self, samples):
        rnds = rand_wrap(2, samples)
        rnds = [[0.0, y, t] for y, t in rnds]
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.boundary_left = rnds

    def calc_boundary_samples_right(self, samples):
        rnds = rand_wrap(2, samples)
        rnds = [[1.0, y, t] for y, t in rnds]
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.boundary_right = rnds

    def calc_boundary_samples_bottom(self, samples):
        rnds = rand_wrap(2, samples)
        rnds = [[x, 0.0, t] for x, t in rnds]
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.boundary_bottom = rnds

    def calc_boundary_samples_top(self, samples):
        rnds = rand_wrap(2, samples)
        rnds = [[x, 1.0, t] for x, t in rnds]
        if self.transform:
            rnds = [self.transform(r) for r in rnds]
        self.boundary_top = rnds

    def boundary_torch(self, device=None):
        all_pts = (
            self.boundary_left
            + self.boundary_right
            + self.boundary_bottom
            + self.boundary_top
        )
        return self.as_torch(all_pts, device=device)

    # --------------------------------------------------------
    # normals
    # --------------------------------------------------------
    def calc_normals(self):
        self.normals_left   = torch.tensor([[-1,0,0]]).repeat(len(self.boundary_left),1)
        self.normals_right  = torch.tensor([[+1,0,0]]).repeat(len(self.boundary_right),1)
        self.normals_bottom = torch.tensor([[0,-1,0]]).repeat(len(self.boundary_bottom),1)
        self.normals_top    = torch.tensor([[0,+1,0]]).repeat(len(self.boundary_top),1)

    def normals_torch(self, device=None):
        normals = torch.cat(
            [self.normals_left,
             self.normals_right,
             self.normals_bottom,
             self.normals_top],
            dim=0
        )
        return normals.to(device) if device else normals
