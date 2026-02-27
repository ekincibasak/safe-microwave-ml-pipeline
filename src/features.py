import numpy as np
from sklearn.decomposition import PCA

def flatten_36x36(X: np.ndarray) -> np.ndarray:
    # X: (N, 36, 36) -> (N, 1296)
    return X.reshape(X.shape[0], -1)

def fit_pca_train_only(X_train_2d: np.ndarray, n_components: int, seed: int):
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_2d)  # FIT ONLY ON TRAIN
    return pca, X_train_pca

def apply_pca(pca: PCA, X_2d: np.ndarray) -> np.ndarray:
    return pca.transform(X_2d)
