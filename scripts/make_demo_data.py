import os
import numpy as np

def make_demo(n_samples=120, seed=42):
    rng = np.random.default_rng(seed)
    # X: (N, 36, 36)
    X = rng.normal(0, 1, size=(n_samples, 36, 36)).astype(np.float32)
    y = rng.integers(0, 2, size=(n_samples,), dtype=np.int64)

    # Add a weak “signal pattern” for class 1
    for i in range(n_samples):
        if y[i] == 1:
            X[i, 10:16, 20:26] += 0.75  # small block bump

    return X, y

if __name__ == "__main__":
    os.makedirs("data_demo", exist_ok=True)
    X, y = make_demo()
    np.savez("data_demo/demo_36x36.npz", X=X, y=y)
    print("Saved: data_demo/demo_36x36.npz", X.shape, y.shape)
