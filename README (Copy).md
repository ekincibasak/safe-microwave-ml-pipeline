# Safe Microwave ML Pipeline (Demo)

A privacy-safe, fully reproducible machine learning pipeline inspired by microwave scattering classification workflows.

This repository demonstrates:

- Structured data processing for 2D scattering matrices (36×36)
- Train-only PCA (no data leakage)
- XGBoost classifier with Optuna hyperparameter optimization
- End-to-end reproducible evaluation
- Clean project structure (configs / scripts / src separation)

⚠️ This repository uses synthetic demo data.
No real patient `.s36p` files or metadata are included.

---

## Project Motivation

Microwave scattering data produces structured matrices that require careful preprocessing before classification.  
This demo replicates the workflow while ensuring:

- No information leakage (PCA fit only on training data)
- Reproducibility via seeded runs
- Config-driven experimentation
- Clean separation between data, features, and model logic

---

## Pipeline Overview

1. Generate synthetic 36×36 demo scattering data
2. Split into train/test
3. Flatten features
4. Fit PCA on training set only
5. Train XGBoost with Optuna optimization
6. Evaluate on held-out test set
7. Save metrics and confusion matrix

---

## Quickstart

```bash
pip install -r requirements.txt

# 1️⃣ Generate synthetic demo data
python scripts/make_demo_data.py

# 2️⃣ Train model (PCA fit on training only)
PYTHONPATH=. python scripts/train_xgb.py --config 


