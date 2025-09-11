# Offline Signature Verification — Siamese Vision Mamba (Tiny)

Siamese network for offline signature verification using a Vision‑Mamba‑Tiny backbone.  
Train on CEDAR (or a folder with the same structure), evaluate, and run quick inference from a saved checkpoint.

> **Checkpoint format:** `checkpoints/siamese_vision_mamba_tiny.pt` is a regular PyTorch *state_dict* (not TorchScript).  
> Keys include: `['model', 'backbone', 'size', 'threshold']`.  
> Load with `torch.load(...); model.load_state_dict(state['model'])`.

---

## 1) Quickstart (Windows / macOS / Linux)

```bash
# 1) Install Anaconda/Miniconda (once)
#    Download from the official site and finish the installer.

# 2) Open "Anaconda Prompt" (Windows) or your terminal (macOS/Linux), then:
conda create -n sigvmamba311 python=3.11 -y
conda activate sigvmamba311

# 3) Go to your project folder (example path — change to yours)
cd "C:\Users\megel\Documents\OFFLINE-SIGNATURE-SIAMESE-VISION-MAMBA"

# 4) (Optional) Open in VS Code
code .

# 5) Install Python packages *inside the conda env*
#    If PyTorch is not pinned in requirements.txt, install it first (CPU example shown):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
#    then the rest:
pip install -r requirements.txt
```

> ✅ **Yes — install requirements from inside Anaconda.**  
> The `conda` env gives you an isolated Python and pip. As long as `conda activate sigvmamba311` is active, `pip install -r requirements.txt` installs into that environment.

**Select the interpreter in VS Code:**  
`Ctrl/Cmd + Shift + P` → “Python: Select Interpreter” → choose `sigvmamba311`.

---

## 2) Expected project layout

Your repo should look like:

```
.
├─ checkpoints/
│  └─ siamese_vision_mamba_tiny.pt
├─ cedar_dataset/
│  ├─ full_org/   # genuine/original signatures
│  └─ full_forg/  # forgeries
├─ cache_preproc/     # will be created automatically
├─ notebooks/
│  └─ siamese_vision_mamba_tiny.ipynb
├─ requirements.txt
└─ readme.md
```

> The notebook auto‑searches for the dataset under `cedar_dataset/` (with `full_org/` and `full_forg/`).  
> If it can’t find it, it will raise a helpful error.

---

## 3) Run the notebook

From your activated env in the project folder:

```bash
conda activate sigvmamba311
code .        # or: jupyter lab
```

Open **`notebooks/siamese_vision_mamba_tiny.ipynb`** and run cells top‑to‑bottom.

---

## 4) Quick inference with the checkpoint

Minimal Python snippet (same logic as the “QUICK INFERENCE” notebook cell):

```python
import torch, torchvision.transforms as T
from PIL import Image

# --- Model definition must match training (SiameseNet with Vision‑Mamba‑Tiny backbone) ---
# The notebook defines these classes; if you run this inside the notebook you already have them.
from your_module import SiameseNet  # or run inside the notebook where SiameseNet is defined

@torch.no_grad()
def load_model_for_infer(ckpt_path, device="cpu"):
    state = torch.load(ckpt_path, map_location=device)
    model = SiameseNet(state["backbone"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, float(state["threshold"]), int(state.get("size", 224))

# simple preprocessing used by the notebook
def binarize_and_crop(pil_img):  # graceful fallback if OpenCV is absent
    return pil_img.convert("L").convert("RGB")

def tf(size):
    return T.Compose([
        T.Lambda(lambda im: binarize_and_crop(im)),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

@torch.no_grad()
def compare_images(model, threshold, img1, img2, size, device="cpu"):
    tr = tf(size)
    x1 = tr(Image.open(img1)).unsqueeze(0).to(device)
    x2 = tr(Image.open(img2)).unsqueeze(0).to(device)
    dist, _, _ = model(x1, x2)          # L2 distance of normalized embeddings
    dist = float(dist.item())
    pred_same = int(dist < threshold)   # 1=same signer, 0=different
    return {"distance": dist, "threshold": threshold, "pred": pred_same}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, th, size = load_model_for_infer("checkpoints/siamese_vision_mamba_tiny.pt", device)
res = compare_images(model, th, "cedar_dataset/full_org/001_01.png",
                     "cedar_dataset/full_forg/001_11.png", size, device)
print(res)  # {'distance': ..., 'threshold': ..., 'pred': 0 or 1}
```

> If you ever see `Not TorchScript: ... constants.pkl ...`, that’s just telling you this file is **not** a TorchScript `.pt`. It’s a *state_dict*. Load it with `torch.load(...)` and `model.load_state_dict(...)` as above (do **not** use `torch.jit.load`).

---

## 5) What each notebook section does (computations explained)

The notebook is structured into clear blocks. Here’s the map:

1. **CONFIG**  
   - Locates the dataset (`cedar_dataset/full_org`, `cedar_dataset/full_forg`).  
   - Sets hyperparameters: `SIZE=224`, `BATCH=16`, `EPOCHS=6`, `LR=2e-4`, `MARGIN=0.7`, `SEED=123`.  
   - Chooses backbone name (default `"mambaout_tiny.in1k"`) with graceful fallbacks via `timm` / torchvision.  
   - Threshold policy for classification from distances: `"youden"` (alternatives: `"maxacc"`, `"eer"`).

2. **DATASET & LOADERS (+ caching, writer‑disjoint split)**  
   - Scans the two folders and groups images by signer.  
   - **Preprocessing:** `binarize_and_crop` (Otsu binarization + tight bounding box when OpenCV is available; otherwise grayscale fallback).  
   - Disk cache of preprocessed grayscale tensors in `cache_preproc/` to speed re‑runs.  
   - Builds **positive pairs** (same signer) and **negative pairs** (different signers) balanced 50/50.  
   - Optional **writer‑disjoint** split (train/val/test signers do not overlap).  
   - PyTorch `DataLoader`s with light geometric/blur augmentation for `train`.

3. **PARTITION SUMMARY + LEAKAGE CHECK**  
   - Prints counts of pairs per split and confirms disjointness between signers (prevents ID leakage).

4. **MODEL (Vision Mamba backbone + Siamese + Contrastive)**  
   - `VisionBackbone` obtains a feature vector `z` (via `timm` Vision‑Mamba‑Tiny if present; falls back to closest available or ResNet18).  
   - `SiameseNet`:  
     - Projects `z → e` through a small MLP head, then L2‑normalizes `e`.  
     - **Distance:** `d = ||e₁ − e₂||₂` (per‑pair L2 on normalized embeddings).  
   - **Contrastive loss** (margin `m`):  
     \\\[
     \mathcal{L} = y \cdot d^2\;+\;(1-y)\cdot \max(0, m - d)^2
     \\\]  
     where `y=1` for *same signer*, `y=0` for *different signer*.

5. **METRIC HELPERS (acc/f1/FAR/FRR/AUC, Youden threshold)**  
   - Computes metrics from distance arrays and labels.  
   - Picks decision threshold either by **Max Accuracy**, **Youden’s J** (`TPR−FPR`), or **EER**.

6. **EER / ROC‑AUC / PR‑AUC + Curves**  
   - Builds ROC/PR curves from `scores = −distance` (smaller distance ⇒ more similar).  
   - Reports AUC/AP and plots curves.

7. **TRAIN / EVAL (freeze→unfreeze + AMP + cosine LR)**  
   - First `FREEZE_EPOCHS` train only the head; then unfreeze the backbone (with a lower LR multiplier).  
   - Uses AdamW + CosineAnnealingLR.  
   - Optional `torch.amp.autocast` mixed precision on CUDA.  
   - Validation picks/bakes the current best threshold according to `THRESH_MODE`.

8. **TRAIN RUN**  
   - Calls `fit(...)`, saves the **best** checkpoint to `checkpoints/siamese_mamba_best_8.pt` (state dict with `model`, `backbone`, `size`, `threshold`).  
   - Prints final **test** accuracy/F1/AUC/FAR/FRR with the frozen best threshold.

9. **PLOTS (train loss + val acc/f1)**  
   - Visualizes training dynamics over epochs.

10. **QUICK INFERENCE (load ckpt + compare_images)**  
    - Utility to load the checkpoint and compare any two images by path.  
    - Returns `distance`, `threshold`, and `pred` (1 = same signer).

11. **SANITY CHECK**  
    - Confirms classes exist and prints the chosen backbone name.

12. **SHOW TWO IMAGES + PREDICTION**  
    - Displays a side‑by‑side of two images and titles with the prediction and distance.

13. **SALIENCY (SmoothGrad) SIMPLE OVERLAY**  
    - Produces rough saliency overlays for each input to visualize which strokes influenced the similarity.

14. **CONFUSION MATRIX + DISTANCE HISTOGRAMS (VAL & TEST)**  
    - Plots confusion matrix at the chosen threshold and histograms of distances for positives vs negatives.

---

## 6) Training tips

- Reduce `BATCH` if you see CUDA OOM.  
- If `timm` cannot find a Vision‑Mamba model, the code falls back to the closest available (or torchvision ResNet18).  
- To strictly reproduce: fix `SEED`, keep writer‑disjoint splits, and avoid changing augmentations.

---

## 7) Common issues

- **“Not TorchScript: constants.pkl not found” when probing the `.pt`**  
  You’re trying to open a *state_dict* with a TorchScript loader. Use:
  ```python
  state = torch.load("checkpoints/siamese_vision_mamba_tiny.pt", map_location="cpu")
  model = SiameseNet(state["backbone"])
  model.load_state_dict(state["model"])
  ```
- **Dataset not found**  
  Make sure `cedar_dataset/full_org` and `cedar_dataset/full_forg` exist (names are flexible; the notebook tries several common aliases).

---

## 8) Reuse in other projects

You can treat the saved checkpoint as a plug‑and‑play verifier:

```python
# returns 1 for same-writer, 0 for different
pred = int(dist < threshold)
```

Export a tiny helper so others can import `compare_images(...)` and `load_model_for_infer(...)` in their pipelines.

---
