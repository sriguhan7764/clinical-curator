# Clinical Curator

> **Multi-label chest X-ray pathology classification powered by 7 deep learning architectures — from custom CNNs to DenseNet-121 with GradCAM interpretability.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is this?

Clinical Curator is a full-stack AI diagnostic tool built for the NIH ChestX-ray14 dataset (112,120 images, 14 disease classes). It covers the full deep learning pipeline across 4 reviews:

| Review | Models Trained | Key Concepts |
|--------|---------------|--------------|
| R1 | ChestMLP, ChestCNN | Custom architectures, grid search, GradCAM |
| R2 | FineTunedResNet-50, CNN-LSTM/GRU | Transfer learning, temporal sequences, attention |
| R3 | ConvAutoencoder, VAE, DCGAN | Unsupervised learning, generative modelling, FID |
| R4 | DenseNet-121, EfficientNet-B2 | Production model, Optuna HPO, ablation study |

The production API and React UI allow clinicians to upload a chest X-ray, run inference across any combination of models, and receive:
- Per-class confidence scores with risk stratification (CRITICAL / CAUTION / NORMAL)
- GradCAM heatmap overlay showing where the model is looking
- Side-by-side multi-architecture comparison
- Downloadable PDF diagnostic report

---

## 14 Disease Classes

```
Atelectasis  ·  Cardiomegaly  ·  Effusion  ·  Infiltration  ·  Mass
Nodule  ·  Pneumonia  ·  Pneumothorax  ·  Consolidation  ·  Edema
Emphysema  ·  Fibrosis  ·  Pleural Thickening  ·  Hernia
```

---

## Repository Layout

```
clinical-curator/
│
├── src/                          # Installable Python package
│   ├── data/
│   │   └── dataset.py            # NIHChestXrayDataset, TemporalPatientDataset,
│   │                             # AdvancedXrayDataset, XrayDataset, get_dataloaders
│   ├── models/
│   │   ├── mlp.py                # ChestMLP — 4-layer deep MLP (R1)
│   │   ├── cnn.py                # ChestCNN — 4-block ConvNet with AdaptivePool (R1)
│   │   ├── pretrained.py         # FeatureExtractor backbone, FineTunedResNet-50 (R2)
│   │   ├── temporal.py           # CNN-RNN Hybrid + Bahdanau/Self/MultiHead attention (R2)
│   │   ├── autoencoder.py        # ConvAutoencoder, VAE with reparameterization (R3)
│   │   ├── gan.py                # DCGAN Generator + SpectralNorm Discriminator (R3)
│   │   └── densenet.py           # DenseNetCXR (production), EfficientNetCXR (R4)
│   ├── training/
│   │   └── trainer.py            # Generic Trainer: BCEWithLogitsLoss, cosine LR,
│   │                             # early stopping, AMP support, checkpoint saving
│   └── utils/
│       ├── gradcam.py            # GradCAM via forward/backward hooks + overlay helper
│       ├── metrics.py            # per_class_auc, per_class_f1, mean_auc
│       └── visualization.py      # ROC curves, F1 heatmap, learning curves (matplotlib)
│
├── scripts/                      # CLI training entry points (argparse)
│   ├── train_r1.py               # MLP + CNN with optional grid search
│   ├── train_r2.py               # ResNet-50 + CNN-RNN hybrid
│   ├── train_r3.py               # AE + VAE + DCGAN with mixed precision
│   └── train_r4.py               # DenseNet-121 with Optuna HPO + ablation
│
├── backend/                      # FastAPI inference server
│   ├── main.py                   # 7 REST endpoints, GradCAM, PDF generation
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                     # React + TypeScript + Vite
│   └── src/
│       ├── pages/                # Landing, Upload, Results, Compare
│       ├── components/ui/        # XrayViewer, PredictionBar, ModelCard, ...
│       ├── hooks/                # usePrediction, useModels
│       ├── utils/format.ts       # getRiskColors, getRiskLevel, formatConfidence
│       └── styles/tokens.css     # CSS custom properties (design tokens)
│
├── notebooks/                    # Original Kaggle notebooks (reference only)
│   ├── review01_mlp_cnn.ipynb
│   ├── review02_pretrained_temporal.ipynb
│   ├── review03_ae_gan.ipynb
│   ├── review04_densenet_e2e.ipynb
│   └── master_combined.ipynb
│
├── configs/models/               # Per-model YAML hyperparameter configs
│   ├── mlp.yaml
│   ├── cnn.yaml
│   ├── pretrained.yaml
│   ├── temporal.yaml
│   ├── autoencoder.yaml
│   └── densenet.yaml
│
├── sample_images/                # Two NIH X-ray samples for demo/testing
├── environment.yml               # Conda environment (Python 3.11 + PyTorch 2.1)
├── setup.py                      # pip install -e . makes src/ importable
└── docker-compose.yml            # One-command full-stack launch
```

---

## Setup

### Option A — Conda (recommended)

```bash
git clone https://github.com/<your-username>/clinical-curator.git
cd clinical-curator

conda env create -f environment.yml
conda activate clinical-curator
pip install -e .
```

### Option B — pip

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[hpo]"
```

### Option C — Docker (full stack)

```bash
docker-compose up --build
# Frontend → http://localhost:3000
# API      → http://localhost:8000/docs
```

---

## Dataset Download

```bash
pip install kagglehub
python - <<'EOF'
import kagglehub
path = kagglehub.dataset_download("nih-chest-xrays/data")
print(f"Downloaded to: {path}")
EOF
```

Or download manually from [Kaggle NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data).

---

## Training

All training scripts accept `--help` for the full argument list.

### Review 1 — MLP & CNN baselines

```bash
python scripts/train_r1.py \
    --data_root /path/to/nih \
    --output_dir outputs/r1 \
    --epochs 10 \
    --subset_frac 0.15 \
    --grid_search          # optional: 2-epoch hyperparameter grid search first
```

### Review 2 — Transfer learning + temporal

```bash
# FineTunedResNet-50 + CNN-LSTM with Bahdanau attention
python scripts/train_r2.py \
    --data_root /path/to/nih \
    --output_dir outputs/r2 \
    --backbone resnet50 \
    --rnn_type LSTM \
    --attn bahdanau
```

### Review 3 — Autoencoders & GAN

```bash
# Trains ConvAE + VAE + DCGAN sequentially
python scripts/train_r3.py \
    --data_root /path/to/nih \
    --output_dir outputs/r3 \
    --ae_epochs 30 \
    --gan_epochs 50
```

### Review 4 — Production DenseNet with Optuna HPO

```bash
python scripts/train_r4.py \
    --data_root /path/to/nih \
    --output_dir outputs/r4 \
    --optuna_trials 20      # hyperparameter search, then full training
```

---

## Running the App

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev    # http://localhost:3000
```

---

## API Reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `GET` | `/api/health` | — | `{ status, models_loaded }` |
| `GET` | `/api/models` | — | List of model metadata |
| `POST` | `/api/predict` | `image` (file), `model_name` | Predictions + GradCAM base64 |
| `POST` | `/api/predict/batch` | `image` (file), `model_names[]` | Array of per-model results |
| `GET` | `/api/latent` | — | 2D t-SNE embeddings for visualisation |
| `POST` | `/api/temporal` | `images[]` (files) | Temporal sequence analysis |
| `POST` | `/api/report/pdf` | `{ prediction_result, patient_id }` | PDF file download |

### Example — single inference

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "image=@sample_images/sample1.png" \
  -F "model_name=densenet121_r4" \
  | python -m json.tool
```

### Example — batch compare

```bash
curl -X POST http://localhost:8000/api/predict/batch \
  -F "image=@sample_images/sample1.png" \
  -F "model_names=mlp_r1" \
  -F "model_names=cnn_r1" \
  -F "model_names=densenet121_r4"
```

### Example — generate PDF report

```bash
curl -X POST http://localhost:8000/api/report/pdf \
  -H "Content-Type: application/json" \
  -d '{"prediction_result": {...}, "patient_id": "PAT-001"}' \
  --output report.pdf
```

---

## Model Architectures

### ChestMLP (R1)
```
Input (150528) → Linear(1024) → BN → ReLU → Dropout(0.4)
               → Linear(512)  → BN → ReLU → Dropout(0.4)
               → Linear(256)  → BN → ReLU → Dropout(0.2)
               → Linear(14)
```

### ChestCNN (R1)
```
Input (3×224×224)
→ [Conv3×3 → BN → ReLU] × 2 → MaxPool   (3→32)
→ [Conv3×3 → BN → ReLU] × 2 → MaxPool   (32→64)
→ [Conv3×3 → BN → ReLU] × 2 → MaxPool   (64→128)
→ [Conv3×3 → BN → ReLU] × 2 → MaxPool   (128→256)
→ AdaptiveAvgPool(4×4) → FC(512) → BN → Dropout → FC(14)
```

### CNN-RNN Hybrid (R2)
```
Input frames → ResNet-50 backbone (frozen)
             → Linear embed → PositionalEncoding
             → BiLSTM (2 layers, hidden=256)
             → BahdanauAttention / SelfAttention / MultiHeadAttention
             → FC(256) → Dropout → FC(14)
```

### DenseNetCXR (R4 — production)
```
DenseNet-121 pretrained (IMAGENET1K_V1)
  Frozen: denseblock1–3
  Trainable: denseblock4, norm5
→ AdaptiveAvgPool(1×1) → Flatten
→ Dropout → Linear(512) → BN → ReLU → Dropout → Linear(14)
```

---

## GradCAM Usage

```python
from src.models.densenet import DenseNetCXR
from src.utils.gradcam import GradCAM
import torch
from PIL import Image
from torchvision import transforms

model = DenseNetCXR()
checkpoint = torch.load("outputs/r4/DenseNet121.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Target the last conv layer
cam = GradCAM(model, layer=model.features.denseblock4)

# Preprocess image
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = Image.open("sample_images/sample1.png").convert("RGB")
x = tf(img).unsqueeze(0).requires_grad_(True)

# Compute heatmap for class 0 (Atelectasis)
GradCAM.disable_inplace_relu(model)
heatmap = cam(x, cls_idx=0)   # ndarray in [0, 1]
overlay = GradCAM.overlay(img_np, heatmap, alpha=0.5)
```

---

## Risk Stratification

All confidence scores are mapped to clinical risk levels for UI display:

| Level | Confidence | CSS Token | Use |
|-------|-----------|-----------|-----|
| `CRITICAL` | ≥ 70% | `--cc-critical-fg` | Immediate specialist escalation recommended |
| `CAUTION` | 40–69% | `--cc-caution-fg` | Secondary review recommended |
| `NORMAL` | < 40% | `--cc-normal-fg` | Routine follow-up |

```ts
// frontend/src/utils/format.ts
const level = getRiskLevel(confidence_pct);   // 'CRITICAL' | 'CAUTION' | 'NORMAL'
const colors = getRiskColors(level);          // { bg, text } CSS variable strings
```

---

## Using the `src` Package in Your Own Code

```python
from src.models.densenet import DenseNetCXR
from src.models.cnn import ChestCNN
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.utils.metrics import evaluate_model, per_class_auc

# Load data
train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
    data_root="/path/to/nih",
    subset_frac=0.15,
    batch_size=32,
)

# Train
model = DenseNetCXR()
trainer = Trainer(model, train_loader, val_loader, pos_weight=pos_weight)
history = trainer.fit(epochs=10, save_path="outputs/densenet.pth")

# Evaluate
y_true, y_prob = evaluate_model(model, test_loader)
aucs = per_class_auc(y_true, y_prob)
for cls, auc in aucs.items():
    print(f"  {cls:<22} {auc:.4f}")
```

---

## Notebooks

The `notebooks/` folder contains the original Kaggle notebooks used during development. They are provided as reference and can be run as-is on Kaggle with GPU. The Python modules in `src/` are the refactored, importable equivalents.

| Notebook | Content |
|----------|---------|
| `review01_mlp_cnn.ipynb` | ChestMLP, ChestCNN, hyperparameter grid, GradCAM |
| `review02_pretrained_temporal.ipynb` | Transfer learning, RNN hybrids, temporal dataset |
| `review03_ae_gan.ipynb` | Autoencoder, VAE, DCGAN with mixed precision |
| `review04_densenet_e2e.ipynb` | DenseNet-121, ablation study, Optuna HPO, latent space |
| `master_combined.ipynb` | End-to-end pipeline combining all reviews |

---

## Dataset Citation

```bibtex
@inproceedings{wang2017chestxray8,
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
               Bagheri, Mohammadhadi and Summers, Ronald M.},
  booktitle = {CVPR},
  year      = {2017}
}
```

---

## Ethical Notice

> This system is for **research and educational purposes only**. It must not be used as a substitute for professional medical diagnosis. All model outputs require review by a qualified radiologist. Performance metrics were measured on the NIH ChestX-ray14 test split and may not generalise to other patient populations or imaging equipment.

---

## License

MIT License — see [LICENSE](LICENSE) for full text.
