import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import json
import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src.seed import seed_everything
from src.data import load_demo_npz
from src.features import (
    flatten_36x36, fit_pca_train_only, apply_pca,
    svd_features, mps_random_embedding
)
from src.model_mlp import MLP

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_features(cfg, X_train, X_test, seed):
    method = cfg["features"].get("method", "pca")
    pca = None

    if method == "pca":
        X_train_2d = flatten_36x36(X_train)
        X_test_2d  = flatten_36x36(X_test)
        if cfg["features"].get("use_pca", True):
            n_comp = int(cfg["features"]["pca_n_components"])
            pca, X_train_2d = fit_pca_train_only(X_train_2d, n_comp, seed)
            X_test_2d = apply_pca(pca, X_test_2d)

    elif method == "svd":
        top_k = int(cfg["svd"]["top_k"])
        X_train_2d = svd_features(X_train, top_k=top_k)
        X_test_2d  = svd_features(X_test, top_k=top_k)

    elif method == "mps":
        X_train_flat = flatten_36x36(X_train)
        X_test_flat  = flatten_36x36(X_test)
        bond_dim = int(cfg["mps"]["bond_dim"])
        out_dim  = int(cfg["mps"]["out_dim"])
        X_train_2d = mps_random_embedding(X_train_flat, bond_dim=bond_dim, out_dim=out_dim, seed=seed)
        X_test_2d  = mps_random_embedding(X_test_flat, bond_dim=bond_dim, out_dim=out_dim, seed=seed)
    else:
        raise ValueError(f"Unknown features.method: {method}")

    return X_train_2d, X_test_2d, pca

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed = int(cfg["seed"])
    seed_everything(seed)

    out_dir = cfg["outputs"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    demo = load_demo_npz(cfg["data"]["demo_npz_path"])
    X = demo.X
    y = demo.y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg["split"]["test_size"]), random_state=seed, stratify=y
    )

    X_train_2d, X_test_2d, pca = make_features(cfg, X_train, X_test, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(X_train_2d, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test_2d, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=int(cfg["mlp"]["batch_size"]), shuffle=True)

    model = MLP(
        in_dim=X_train_2d.shape[1],
        hidden_dim=int(cfg["mlp"]["hidden_dim"]),
        dropout=float(cfg["mlp"]["dropout"]),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["mlp"]["lr"]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(int(cfg["mlp"]["epochs"])):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device)).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }

    # Save
    torch.save(model.state_dict(), os.path.join(out_dir, "mlp_state_dict.pt"))
    joblib.dump({"pca": pca, "cfg": cfg}, os.path.join(out_dir, "mlp_preproc.joblib"))
    with open(os.path.join(out_dir, "mlp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:", os.path.join(out_dir, "mlp_state_dict.pt"))
    print("Metrics:", metrics)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
