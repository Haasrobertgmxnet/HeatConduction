# sampling_service.py

import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class SamplingConfig:
    n_interior_samples: int = 3000
    n_initial_samples: int = 100
    n_boundary_samples: int = 100

# --------------------------------------------------
# Optional helper: affine transform builder
# --------------------------------------------------
def make_affine_transform(a, B):
    """
    Returns a callable v ↦ a + B @ v
    Works vectorized on (N,3).
    """
    a = np.asarray(a).reshape(1, -1)     # (1,3)
    B = np.asarray(B).reshape(3, 3)      # (3,3)

    def f(v):
        # v can be (3,) or (N,3)
        v = np.asarray(v)
        return a + v @ B.T

    return f

def _rand_wrap():
    r1 = np.random.rand()
    r2 = np.random.rand()
    return 0.5*(1+r1) if r1 > r2 else 0.5*(1-r2)

def rand_wrap(dims, samples):
    return [[_rand_wrap() for _ in range(dims)] for _ in range(samples)]


class SamplingService:
    """
    SamplingService with MODE:
        mode="uniform" → uniform interior sampling
        mode="biased"  → oversample region near t=0 for PDE stability
    """
    def __init__(self, config= None, transform=None, mode="uniform", frac_near0=0.3, t_eps=0.02):
        self.config: SamplingConfig = config
        self.transform = transform
        self.mode = mode
        self.frac_near0 = frac_near0
        self.t_eps = t_eps

    # ---------------------------------
    # Internal helper
    # ---------------------------------
    def _as_torch(self, data, device=None, dtype=torch.float32):
        t = torch.tensor(data, dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t

    # ---------------------------------
    # Interior sampling
    # ---------------------------------
    def calc_interior_samples(self):
        rnds = rand_wrap(3, self.config.n_interior_samples)
        if self.transform:
            rnds = self.transform(np.array(rnds))
            # rnds = [self.transform(r) for r in rnds]
        self.interior_samples = rnds

    def get_interior_samples_uniform(self, device=None):
        return self._as_torch(self.interior_samples, device=device)

    def get_interior_samples_biased(self, device=None):
        """
        Oversample region t <= t_eps by fraction frac_near0.
        """
        n_samples = self.config.n_interior_samples
        xyt = torch.tensor(self.interior_samples)
        mask = xyt[:, 2] <= self.t_eps

        idx_near = torch.where(mask)[0]
        idx_all  = torch.arange(len(xyt))

        n_near = int(n_samples * self.frac_near0)

        if len(idx_near) > 0:
            perm_near = idx_near[torch.randperm(len(idx_near))[:min(n_near,len(idx_near))]]
        else:
            perm_near = torch.tensor([], dtype=torch.long)

        rest = n_samples - len(perm_near)
        perm_rest = idx_all[torch.randperm(len(idx_all))[:rest]]

        idx = torch.cat([perm_near, perm_rest])
        xyt_sel = xyt[idx]

        if device:
            xyt_sel = xyt_sel.to(device)

        return xyt_sel.float()

    def get_interior_samples_torch(self, device=None):
        """
        Main method used by training pipeline.
        Automatically applies chosen mode.
        """
        n_samples = self.config.n_interior_samples
        if self.mode == "uniform":
            return self.get_interior_samples_uniform(device)

        elif self.mode == "biased":
            if n_samples is None:
                raise ValueError("n_samples required for biased sampling")
            return self.get_interior_samples_biased(device=device)

        else:
            raise ValueError(f"Unknown sampling mode: {self.mode}")

    # ---------------------------------
    # Initial condition samples: t=0
    # ---------------------------------
    def calc_initial_samples(self):
        rnds = rand_wrap(2, self.config.n_initial_samples)
        rnds = [[x, y, 0.0] for x, y in rnds]
        if self.transform:
            rnds = self.transform(np.array(rnds))
            # rnds = [self.transform(r) for r in rnds]
        self.initial_samples = rnds

    def get_initial_samples_torch(self, device=None):
        return self._as_torch(self.initial_samples, device=device)

    # ---------------------------------
    # Boundary samples
    # ---------------------------------
    def calc_boundary_samples_left(self):
        rnds = rand_wrap(2, self.config.n_boundary_samples)
        rnds = [[0.0, y, t] for y, t in rnds]
        rnds = np.array(rnds, dtype=float)
        if self.transform:
            rnds = self.transform(rnds)
        self.boundary_left = rnds

    def calc_boundary_samples_right(self):
        rnds = rand_wrap(2, self.config.n_boundary_samples)
        rnds = [[1.0, y, t] for y, t in rnds]
        rnds = np.array(rnds, dtype=float)
        if self.transform:
            rnds = self.transform(rnds)
        self.boundary_right = rnds

    def calc_boundary_samples_bottom(self):
        rnds = rand_wrap(2, self.config.n_boundary_samples)
        rnds = [[x, 0.0, t] for x, t in rnds]
        rnds = np.array(rnds, dtype=float)
        if self.transform:
            rnds = self.transform(rnds)
        self.boundary_bottom = rnds

    def calc_boundary_samples_top(self):
        rnds = rand_wrap(2, self.config.n_boundary_samples)
        rnds = [[x, 1.0, t] for x, t in rnds]
        rnds = np.array(rnds, dtype=float)
        if self.transform:
            rnds = self.transform(rnds)
        self.boundary_top = rnds

    def get_boundary_samples_torch(self, device=None):
        pts = np.concatenate(
            [
                np.asarray(self.boundary_left),
                np.asarray(self.boundary_right),
                np.asarray(self.boundary_bottom),
                np.asarray(self.boundary_top),
            ],
            axis=0
        )
        # print(f"Total boundary samples: {len(pts)}")
        return self._as_torch(pts, device=device)

    # ---------------------------------
    # Normals
    # ---------------------------------
    def calc_normals(self):
        self.normals_left   = torch.tensor([[-1,0,0]]).repeat(len(self.boundary_left),1)
        self.normals_right  = torch.tensor([[+1,0,0]]).repeat(len(self.boundary_right),1)
        self.normals_bottom = torch.tensor([[0,-1,0]]).repeat(len(self.boundary_bottom),1)
        self.normals_top    = torch.tensor([[0,+1,0]]).repeat(len(self.boundary_top),1)

    def get_normals_torch(self, device=None):
        normals = torch.cat([
            self.normals_left,
            self.normals_right,
            self.normals_bottom,
            self.normals_top
        ], dim=0)
        if device:
            normals = normals.to(device)
        return normals

    def calc(self, cfg: SamplingConfig):
        self.calc_interior_samples(cfg.n_interior_samples)
        self.calc_initial_samples(cfg.n_initial_samples)
        self.calc_boundary_samples_left(cfg.n_boundary_samples)
        self.calc_boundary_samples_right(cfg.n_boundary_samples)
        self.calc_boundary_samples_bottom(cfg.n_boundary_samples)
        self.calc_boundary_samples_top(cfg.n_boundary_samples)
        self.calc_normals()

    def print_interior_samples(self):
        print("Interior samples:")
        for s in self.interior_samples:
            print(s)

    def print_initial_samples(self):
        print("Initial samples:")
        for s in self.initial_samples:
            print(s)

    def print_boundary_left(self):
        print("Boundary left samples:")
        for s in self.boundary_left:
            print(s)

    def print_boundary_right(self):
        print("Boundary right samples:")
        for s in self.boundary_right:
            print(s)

    def print_boundary_bottom(self):
        print("Boundary bottom samples:")
        for s in self.boundary_bottom:
            print(s)

    def print_boundary_top(self):
        print("Boundary top samples:")
        for s in self.boundary_top:
            print(s)

    def print_samples_min_max(self):
        for [key, samples] in ({"interior_samples":self.interior_samples,
                        "initial_samples": self.initial_samples,
                        "boundary_left": self.boundary_left,
                        "boundary_right": self.boundary_right,
                        "boundary_bottom": self.boundary_bottom,
                        "boundary_top": self.boundary_top}).items():
            if len(samples) == 0:
                print("Warning: One of the sample sets is empty!")
                continue
            mins = samples.min(axis=0)
            maxs = samples.max(axis=0)
            print(key)
            print("Samples min/max:")
            print(f" x: {mins[0]:.4f} / {maxs[0]:.4f}")
            print(f" y: {mins[1]:.4f} / {maxs[1]:.4f}")
            print(f" t: {mins[2]:.4f} / {maxs[2]:.4f}")

    def get_samples(self, heat_problem):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {device}")

        # Interior Points
        self.calc_interior_samples()
        xyt_interior = self.get_interior_samples_torch(device=device).float()

        # Initial Condition (t = 0)
        self.calc_initial_samples()
        Xyt_ic = self.get_initial_samples_torch(device=device).float()

        # initial_u(x,y) aus IBVPData verwenden
        x_ic = Xyt_ic[:, 0:1]
        y_ic = Xyt_ic[:, 1:2]
        u_ic_val = heat_problem.initial_u(x_ic, y_ic)  # (N,1) oder (N,)
        if u_ic_val.dim() == 1:
            u_ic_val = u_ic_val.unsqueeze(1)
        u_ic = u_ic_val.to(device)

        # Continuity-Punkte: gleiche (x,y), aber t = eps
        eps_time = 1e-3
        Xyt_ic_eps = Xyt_ic.clone()
        Xyt_ic_eps[:, 2] = eps_time

        # Boundary
        self.calc_boundary_samples_left()
        self.calc_boundary_samples_right()
        self.calc_boundary_samples_bottom()
        self.calc_boundary_samples_top()

        xyt_boundary = self.get_boundary_samples_torch(device=device).float()
        #samp.print_boundary_left()

        # samp.print_samples_min_max()

        self.calc_normals()
        normals = self.get_normals_torch(device=device).float()

        return xyt_interior, Xyt_ic, xyt_boundary, normals, u_ic, Xyt_ic_eps
        

