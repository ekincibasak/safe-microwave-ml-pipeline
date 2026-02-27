import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from neuralop.models import FNO
from mask_utils import (
    read_sparam_as_tensor,
    generate_mask_from_location,
    parse_file_info
)


# ------------------------------
# 📁 Paths
# ------------------------------
test_dir = rtest_dir = r"C:\Users\RTX4090\Desktop\currently_working\basak_in_lab\eight_report\reading_location\data\test_healty"
model_path = r"C:\Users\RTX4090\Desktop\currently_working\basak_in_lab\nineth_report\tumor_50localization_fno.pt"
save_dir = r"C:\Users\RTX4090\Desktop\the_code_that_used_for_location\30_07_2025\results"
os.makedirs(save_dir, exist_ok=True)

# ------------------------------
# ⚙️ Load model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fno = FNO(
    in_channels=38,
    out_channels=1,
    n_modes=(16, 16),
    hidden_channels=64,
    n_layers=4,
    padding=9
).to(device)

fno.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
fno.eval()

# ------------------------------
# 🔍 Test each healthy patient
# ------------------------------
test_files = [f for f in os.listdir(test_dir) if f.endswith(".s36p")]

for fname in test_files:
    path = os.path.join(test_dir, fname)

    # Read S-parameters
    sparam = read_sparam_as_tensor(path)  # (38, 36, 36)
    sparam = np.transpose(sparam, (1, 2, 0))  # → (36, 36, 38)
    sparam_tensor = torch.tensor(sparam, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 38, 36, 36)

    # Predict mask
    with torch.no_grad():
        pred_mask = fno(sparam_tensor).cpu().squeeze().numpy()

    # Use an all-zero mask for healthy patients
    true_mask = np.zeros((36, 36), dtype=np.float32)

    # --- Compute metrics ---
    threshold = 0.5
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    true_binary = (true_mask >= 0.5).astype(np.uint8)

    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()

    # Dice Score
    intersection = np.sum(pred_flat * true_flat)
    dice_score = (2. * intersection) / (np.sum(pred_flat) + np.sum(true_flat) + 1e-8)

    # IoU
    union = np.sum((pred_flat + true_flat) > 0)
    iou = intersection / (union + 1e-8)

    # Pixel Accuracy
    accuracy = np.sum(pred_flat == true_flat) / len(pred_flat)

    # MSE
    mse = np.mean((pred_mask - true_mask) ** 2)

    # Print metrics
    print(f"🟢 File: {fname} (Healthy)")
    print(f" Dice Score     : {dice_score:.4f}")
    print(f" IoU            : {iou:.4f}")
    print(f" Pixel Accuracy : {accuracy:.4f}")
    print(f" MSE            : {mse:.6f}")
    print("-" * 50)

    # Optional: save the prediction mask as an image
    plt.imshow(pred_mask, cmap='hot')
    plt.title(f"Predicted Mask - {fname}", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname.replace('.s36p', '')}_pred.png"), dpi=300)
    plt.close()
