import os
import yaml
import joblib
import optuna
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.seed import seed_everything
from src.data import load_demo_npz
from src.features import flatten_36x36, fit_pca_train_only, apply_pca
from src.model_xgb import make_xgb

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

    # A) Flatten
    X_train_2d = flatten_36x36(X_train)
    X_test_2d = flatten_36x36(X_test)

    # PCA fit on TRAIN ONLY
    pca = None
    if cfg["features"]["use_pca"]:
        n_comp = int(cfg["features"]["pca_n_components"])
        pca, X_train_2d = fit_pca_train_only(X_train_2d, n_comp, seed)
        X_test_2d = apply_pca(pca, X_test_2d)

    scoring = cfg["optuna"]["scoring"]
    cv_folds = int(cfg["optuna"]["cv_folds"])

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

        model = make_xgb(
            params=params,
            seed=seed,
            nthread=int(cfg["xgb"]["nthread"]),
            tree_method=cfg["xgb"]["tree_method"],
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train_2d, y_train, cv=cv, scoring=scoring)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=int(cfg["optuna"]["n_trials"]))

    best_params = study.best_trial.params
    best_model = make_xgb(
        params=best_params,
        seed=seed,
        nthread=int(cfg["xgb"]["nthread"]),
        tree_method=cfg["xgb"]["tree_method"],
    )
    best_model.fit(X_train_2d, y_train)

    joblib.dump({"model": best_model, "pca": pca, "cfg": cfg}, os.path.join(out_dir, "artifact.joblib"))
    joblib.dump(study.best_trial.params, os.path.join(out_dir, "best_params.joblib"))
    print("Saved artifact to:", os.path.join(out_dir, "artifact.joblib"))
    print("Best CV score:", study.best_value)
    print("Best params:", best_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
