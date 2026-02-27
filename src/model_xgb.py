from xgboost import XGBClassifier

def make_xgb(params: dict, seed: int, nthread: int = 1, tree_method: str = "hist") -> XGBClassifier:
    return XGBClassifier(
        **params,
        random_state=seed,
        nthread=nthread,
        tree_method=tree_method,
        eval_metric="logloss",
        use_label_encoder=False
    )
