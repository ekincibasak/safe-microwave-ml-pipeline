from dataclasses import dataclass
import numpy as np

@dataclass
class DemoData:
    X: np.ndarray  # (N, 36, 36)
    y: np.ndarray  # (N,)

def load_demo_npz(npz_path: str) -> DemoData:
    arr = np.load(npz_path)
    X = arr["X"].astype(np.float32)
    y = arr["y"].astype(np.int64)
    if X.ndim != 3 or X.shape[1:] != (36, 36):
        raise ValueError(f"Expected X shape (N, 36, 36), got {X.shape}")
    return DemoData(X=X, y=y)
