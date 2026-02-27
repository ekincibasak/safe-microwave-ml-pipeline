# Safe Microwave ML Pipeline (Demo)  
   
A privacy-safe, fully reproducible ML pipeline inspired by microwave scattering classification workflows.  
   
This repository is designed as a **public-safe twin** of a real microwave-scattering project: it runs end-to-end on **synthetic demo matrices** (36×36) while demonstrating the same engineering and ML practices used in real medical / sensing pipelines.  
   
✅ No real patient `.s36p` files.    
✅ No patient identifiers or metadata.    
✅ Fully runnable and reproducible.  
   
---  
   
## What this repo demonstrates  
   
- **Structured processing** of 2D scattering-style matrices (36×36)  
- **Leakage-safe preprocessing** (PCA fit on training only)  
- Multiple **feature extraction** options:  
  - PCA baseline  
  - SVD low-rank descriptors  
  - Tensor-network-inspired (MPS-style) embedding  
- Two **model baselines**:  
  - XGBoost (Optuna-optimized)  
  - Basic neural network (MLP, PyTorch)  
- **Config-driven experiments** (`.yaml`)  
- Saved **artifacts + metrics + plots** for reporting  
   
---  
   
## Pipeline overview  
   
1. Generate synthetic scattering-like demo data (`36×36`)  
2. Split into train/test (stratified)  
3. Extract features (choose via config):  
   - `pca`: flatten → train-only PCA → feature vector  
   - `svd`: top-k singular values + low-rank summary features  
   - `mps`: tensor-network-inspired embedding (compact vector)  
4. Train model:  
   - XGBoost + Optuna search **or**  
   - MLP baseline (PyTorch)  
5. Evaluate and save:  
   - `metrics.json`  
   - `confusion_matrix.png`  
   - trained artifacts (`artifact.joblib`, model weights)  
   
---  
   
## Quickstart  
   
```bash  
pip install -r requirements.txt  
   
# 1) Generate synthetic demo data  
python scripts/make_demo_data.py  
