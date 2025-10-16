from functools import wraps
import numpy as np
import torch

# This is from ChatGPT : 
def numpy_torch_compatible(func=None, *, prefer=None):
    def _decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # heuristik: erstes arg ist "self"/"cls" wenn es ein Objekt mit __dict__ ist
            is_method = len(args) > 0 and hasattr(args[0], "__dict__") and \
                        not isinstance(args[0], (np.ndarray, torch.Tensor))
            start_idx = 1 if is_method else 0

            # scan inputs um device / ob tensor/numpy vorhanden ist
            any_torch = False
            any_numpy = False
            target_device = None

            def scan(x):
                nonlocal any_torch, any_numpy, target_device
                if isinstance(x, torch.Tensor):
                    any_torch = True
                    if target_device is None:
                        target_device = x.device
                elif isinstance(x, np.ndarray):
                    any_numpy = True
                elif isinstance(x, (list, tuple)):
                    for e in x:
                        scan(e)
                elif isinstance(x, dict):
                    for e in x.values():
                        scan(e)
                # sonst: python scalar etc. ignorieren

            for a in args[start_idx:]:
                scan(a)
            for v in kwargs.values():
                scan(v)

            if prefer == 'torch':
                output_torch = True
            elif prefer == 'numpy':
                output_torch = False
            else:
                output_torch = any_torch  # wenn irgendein torch input da ist -> torch out

            # konvertierer: numpy -> torch (rekursiv)
            def to_torch(x):
                if isinstance(x, np.ndarray):
                    t = torch.as_tensor(x)  # guter allgemeiner Weg
                    if target_device is not None:
                        t = t.to(target_device)
                    return t
                elif isinstance(x, torch.Tensor):
                    return x
                elif isinstance(x, (list, tuple)):
                    cls = list if isinstance(x, list) else tuple
                    return cls(to_torch(e) for e in x)
                elif isinstance(x, dict):
                    return {k: to_torch(v) for k, v in x.items()}
                else:
                    return x  # python scalar, object, etc.

            # Inputs konvertieren (self überspringen)
            new_args = list(args)
            for i in range(start_idx, len(args)):
                new_args[i] = to_torch(new_args[i])
            new_kwargs = {k: to_torch(v) for k, v in kwargs.items()}

            # Funktion ausführen (intern mit torch)
            result = f(*new_args, **new_kwargs)

            # Rückkonvertierer: torch -> numpy (rekursiv) falls output_torch==False
            def to_numpy_if_requested(x):
                if isinstance(x, torch.Tensor):
                    if output_torch:
                        return x
                    return x.detach().cpu().numpy()
                elif isinstance(x, (list, tuple)):
                    cls = list if isinstance(x, list) else tuple
                    return cls(to_numpy_if_requested(e) for e in x)
                elif isinstance(x, dict):
                    return {k: to_numpy_if_requested(v) for k, v in x.items()}
                else:
                    return x

            return to_numpy_if_requested(result)

        return wrapper

    if func is None:
        return _decorator
    else:
        return _decorator(func)

class GaussKernel:
    def __init__(self, center_x, center_y, radius, strength = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t = None):
        squared_distance = torch.tensor((x - self.center_x)**2 + (y - self.center_y)**2).detach().clone()
        # squared_distance = ((x - self.center_x)**2 + (y - self.center_y)**2).detach().clone()
        return self.strength * torch.exp(-squared_distance / (2 * self.radius**2))

class ConstantFunc:
    def __init__(self, strength = 1.0):
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t = None):
        if isinstance(x, torch.Tensor):
            return self.strength * torch.ones_like(x)
        return torch.Tensor([self.strength])
