import torch as th
import numpy as np

# Cache device to avoid repeated lookups
_cuda_device = None
_cpu_device = th.device('cpu')

def get_device(use_cuda):
    """Get cached device for efficiency."""
    global _cuda_device
    if use_cuda:
        if _cuda_device is None:
            _cuda_device = th.device('cuda')
        return _cuda_device
    return _cpu_device

def to_tensor_var(x, use_cuda=True, dtype="float"):
    """Optimized tensor creation - creates on CPU then moves to GPU in one transfer."""
    device = get_device(use_cuda)

    if dtype == "float":
        if isinstance(x, np.ndarray):
            tensor = th.from_numpy(x.astype(np.float32)).to(device)
        else:
            tensor = th.tensor(x, dtype=th.float32, device=device)
        return tensor
    elif dtype == "long":
        if isinstance(x, np.ndarray):
            tensor = th.from_numpy(x.astype(np.int64)).to(device)
        else:
            tensor = th.tensor(x, dtype=th.int64, device=device)
        return tensor
    elif dtype == "byte":
        if isinstance(x, np.ndarray):
            tensor = th.from_numpy(x.astype(np.uint8)).to(device)
        else:
            tensor = th.tensor(x, dtype=th.uint8, device=device)
        return tensor
    else:
        if isinstance(x, np.ndarray):
            tensor = th.from_numpy(x.astype(np.float32)).to(device)
        else:
            tensor = th.tensor(x, dtype=th.float32, device=device)
        return tensor


