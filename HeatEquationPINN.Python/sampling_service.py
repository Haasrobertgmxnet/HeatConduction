from enum import Enum
import numpy as np
import torch

def _rand_wrap():
    res1 = np.random.rand()
    res2 = np.random.rand()
    if res1 > res2:
        return 0.5 * (1.0 + res1)
    return 0.5 * (1.0 - res2)

def rand_wrap(dims: int, samples: int):
    return [[_rand_wrap() for _ in range(dims)] for _ in range(samples)]

class SamplingService:
    def __init__(self, transform=None):
        self.transform = transform

    # ---------- Helper ----------
    def _as_torch(self, data, device=None, dtype=torch.float32):
        t = torch.tensor(data, dtype=dtype)
        if device is not None:
            t = t.to(device)
        return t

    # ---------- Interior ----------
    def calc_interior_samples(self, samples: int):
        rnds = rand_wrap(3, samples)
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.interior_samples = rnds  # Liste

    def get_interior_samples(self):
        return self.interior_samples

    def get_interior_samples_torch(self, device=None):
        return self._as_torch(self.interior_samples, device=device)

    def print_interior_samples(self):
        print("print_interior_samples")
        print(f"Random numbers: {self.interior_samples}")

    def get_interior_samples_biased_t0(self, n_samples, t_eps=0.02, frac_near0=0.3, device=None):
        xyt = torch.tensor(self.interior_samples)
        mask = xyt[:,2] <= t_eps

        idx_near = torch.where(mask)[0]
        idx_all = torch.arange(len(xyt))

        n_near = int(n_samples * frac_near0)
        take_near = idx_near[torch.randperm(len(idx_near))[:n_near]]

        rest = n_samples - len(take_near)
        take_rest = idx_all[torch.randperm(len(idx_all))[:rest]]

        idx = torch.cat([take_near, take_rest])
        xyt_sel = xyt[idx]

        if device:
            xyt_sel = xyt_sel.to(device)

        return xyt_sel.float()


    # ---------- Initial (t=0) ----------
    def calc_initial_samples(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[x, y, 0.0] for x, y in rnds]
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.initial_samples = rnds

    def get_initial_samples(self):
        return torch.tensor(self.initial_samples)

    def get_initial_samples_torch(self, device=None):
        return self._as_torch(self.initial_samples, device=device)

    def print_initial_samples(self):
        print("print_intitial_samples")
        print(f"Random numbers: {self.initial_samples}")

    # ---------- Boundary ----------
    def calc_boundary_samples_left(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[0.0, y, t] for y, t in rnds]
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.boundary_samples_left = rnds

    def calc_boundary_samples_right(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[1.0, y, t] for y, t in rnds]
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.boundary_samples_right = rnds

    def calc_boundary_samples_bottom(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[x, 0.0, t] for x, t in rnds]
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.boundary_samples_bottom = rnds

    def calc_boundary_samples_top(self, samples: int):
        rnds = rand_wrap(2, samples)
        rnds = [[x, 1.0, t] for x, t in rnds]
        if self.transform is not None:
            rnds = [self.transform(it) for it in rnds]
        self.boundary_samples_top = rnds

    def get_boundary_samples(self):
        return (self.boundary_samples_left,
                self.boundary_samples_right,
                self.boundary_samples_bottom,
                self.boundary_samples_top)

    def get_boundary_samples_torch(self, device=None):
        all_pts = (self.boundary_samples_left
                   + self.boundary_samples_right
                   + self.boundary_samples_bottom
                   + self.boundary_samples_top)
        return self._as_torch(all_pts, device=device)

    # ---------- Normals ----------
    def calc_normals(self):
        n_left = len(self.boundary_samples_left)
        self.normals_left = torch.tensor([[-1.0, 0.0, 0.0]]).repeat(n_left, 1)
        n_right = len(self.boundary_samples_right)
        self.normals_right = torch.tensor([[1.0, 0.0, 0.0]]).repeat(n_right, 1)
        n_bottom = len(self.boundary_samples_bottom)
        self.normals_bottom = torch.tensor([[0.0, -1.0, 0.0]]).repeat(n_bottom, 1)
        n_top = len(self.boundary_samples_top)
        self.normals_top = torch.tensor([[0.0, 1.0, 0.0]]).repeat(n_top, 1)

    def get_normals(self):
        return (self.normals_left,
                self.normals_right,
                self.normals_bottom,
                self.normals_top)

    def get_normals_torch(self, device=None):
        normals = torch.cat(
            [self.normals_left,
             self.normals_right,
             self.normals_bottom,
             self.normals_top],
            dim=0
        )
        if device is not None:
            normals = normals.to(device)
        return normals

    # ---------- Debug ----------
    def print_boundary_samples(self):
        print("print_boundary_samples")
        print(f"boundary_samples_left: {self.boundary_samples_left}")
        print(f"boundary_samples_right: {self.boundary_samples_right}")
        print(f"boundary_samples_bottom: {self.boundary_samples_bottom}")
        print(f"boundary_samples_top: {self.boundary_samples_top}")

    def print_normals(self):
        print("print_normals")
        print(f"normals_left: {self.normals_left}")
        print(f"normals_right: {self.normals_right}")
        print(f"normals_bottom: {self.normals_bottom}")
        print(f"normals_top: {self.normals_top}")
