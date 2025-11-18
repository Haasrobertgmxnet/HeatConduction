from functools import wraps
import numpy as np
import torch

# Decorator that allows functions to accept both numpy arrays and torch tensors
# It converts all inputs to torch tensors internally, executes the function,
# and optionally converts the output back to numpy depending on inferred or preferred mode.
def numpy_torch_compatible(func=None, *, prefer=None):
    def _decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Check whether the function is a method (first argument is an object instance)
            is_method = len(args) > 0 and hasattr(args[0], "__dict__") and \
                        not isinstance(args[0], (np.ndarray, torch.Tensor))
            start_idx = 1 if is_method else 0

            # Track what types of inputs are present
            any_torch = False
            any_numpy = False
            target_device = None

            # Recursively scan arguments to detect numpy or torch usage
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
                # Other objects are ignored

            # Apply scanner to all positional and keyword arguments except possibly self
            for a in args[start_idx:]:
                scan(a)
            for v in kwargs.values():
                scan(v)

            # Determine whether the output should be torch or numpy
            if prefer == 'torch':
                output_torch = True
            elif prefer == 'numpy':
                output_torch = False
            else:
                output_torch = any_torch  # if any torch input was found, output torch

            # Convert numpy inputs to torch recursively
            def to_torch(x):
                if isinstance(x, np.ndarray):
                    t = torch.as_tensor(x)
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
                    return x  # Python scalar or other object types

            # Convert arguments to torch before calling the wrapped function
            new_args = list(args)
            for i in range(start_idx, len(args)):
                new_args[i] = to_torch(new_args[i])
            new_kwargs = {k: to_torch(v) for k, v in kwargs.items()}

            # Call the wrapped function in torch mode
            result = f(*new_args, **new_kwargs)

            # Convert torch outputs to numpy if needed
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

# Kernel representing a Gaussian function centered at a given point
class GaussKernel:
    def __init__(self, center_x, center_y, radius, strength = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t = None):
        # Compute squared distance from center
        squared_distance = torch.tensor((x - self.center_x)**2 + (y - self.center_y)**2).detach().clone()
        # Return Gaussian value
        return self.strength * torch.exp(-squared_distance / (2 * self.radius**2))

# Kernel with compact support Gaussian profile
class CompactGaussKernel:
    def __init__(self, center_x, center_y, radius, strength=1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t=None):
        # Compute distance from center
        squared_distance = (x - self.center_x)**2 + (y - self.center_y)**2
        r = torch.sqrt(squared_distance)
        # Mask points outside the radius
        mask = r <= self.radius
        # Assign zero outside radius and Gaussian inside
        value = torch.zeros_like(r)
        value[mask] = self.strength * torch.exp(-squared_distance[mask] / (2 * self.radius**2))
        return value

# Kernel with compact support cylindrical profile (flat inside radius)
class CompactCylindricalKernel:
    def __init__(self, center_x, center_y, radius, strength=1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t=None):
        # Compute distance from center
        squared_distance = (x - self.center_x)**2 + (y - self.center_y)**2
        r = torch.sqrt(squared_distance)
        # Mask region inside the radius
        mask = r <= self.radius
        # Assign constant value inside radius, zero outside
        value = torch.zeros_like(r)
        value[mask] = self.strength
        return value

# Function that always returns a constant field
class ConstantFunc:
    def __init__(self, strength = 1.0):
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t = None):
        # If x is a tensor, return a tensor of ones scaled by strength
        if isinstance(x, torch.Tensor):
            return self.strength * torch.ones_like(x)
        # If numpy or scalar input, return scalar wrapped in a torch tensor
        return torch.Tensor([self.strength])

class ModeFunction:
    def __init__(self, strength = 1.0):
        self.strength = strength

    @numpy_torch_compatible
    def evaluate(self, x, y, t=None):
        k= 0
        mode = [0.48503638, 0.1087893,  0.04307502, 0.0239897, 0.01570866][k]
        scal = [0.47979843,0.19595146,0.10821043,0.0736955,0.05570078][k]
        mode_x = mode
        mode_y = mode
        scal_x = scal
        scal_y = scal
        gamma = 0.5
        phi_x = np.sin(mode_x * x) + (mode_x/gamma)*np.cos(mode_x * x)
        phi_y = np.sin(mode_y * y) + (mode_y/gamma)*np.cos(mode_y * y)

        return (scal_x * phi_x) * (scal_y * phi_y) * self.strength