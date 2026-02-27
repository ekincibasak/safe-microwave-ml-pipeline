import os
import yaml
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from src.seed import seed_everything
from src.data import load_demo_npz
from src.features import flatten_36x36, apply_pca
from src.metrics import compute_metrics, save_json

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    seed = int(cfg["seed"])
    seed_everything(seed)

    out_dir = cfg["outputs"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    artifact = joblib.load(os.path.join(out_dir, "artifact.joblib"))
    model = artifact["model"]
    pca = artifact["pca"]

    demo = load_demo_npz(cfg["data"]["demo_npz_path"])
    X = demo.X
    y = demo.y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg["split"]["test_size"]), random_state=seed, stratify=y
    )

    X_test_2d = flatten_36x36(X_test)
    if pca is not None:
        X_test_2d = apply_pca(pca, X_test_2d)

    y_pred = model.predict(X_test_2d)
    y_proba = model.predict_proba(X_test_2d)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba=y_proba)
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    print("Saved metrics to:", os.path.join(out_dir, "metrics.json"))

    # Confusion matrix plot (simple)
    cm = metrics["confusion_matrix"]
    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.figure()
    plt.imshow([[cm["tn"], cm["fp"]],[cm["fn"], cm["tp"]]])
    plt.title("Confusion Matrix (Demo)")
    plt.xticks([0,1], ["Pred 0", "Pred 1"])
    plt.yticks([0,1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(np.array([[cm["tn"], cm["fp"]],[cm["fn"], cm["tp"]]])):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved figure to:", fig_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
