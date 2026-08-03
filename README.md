# FedTrace-FM

## Research Goal
FedTrace-FM investigates traceability and attribution in federated learning
systems involving foundation models — studying how malicious or corrupted
client contributions can be detected, attributed to their source, and
repaired within a federated training pipeline.

## Environment
- OS: Windows 11
- IDE: VS Code
- Python: 3.12 (local virtual environment: `.venv`)
- Deep learning framework: PyTorch 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (~8 GB VRAM)
- Execution: fully local (no Colab, no Google Drive); CUDA used when
  available, with automatic CPU fallback

## Current Development Stage
CIFAR-100 data inspection pipeline implemented. No model, federated
learning, attack, or attribution logic has been implemented yet.

## Data Pipeline Usage

Configuration lives in [configs/cifar100_data.yaml](configs/cifar100_data.yaml).
CIFAR-100 is downloaded automatically into `./data` on first run.

Run the dataset inspection from the project root (PowerShell):

```powershell
.venv\Scripts\python -m src.data.inspect_dataset
```

This prints train/test sizes, class count, image shape, and label range,
verifies all 100 classes are present with valid labels, and writes to
`results/data_inspection/`:
- `cifar100_samples.png` — 5x5 grid of unnormalized sample images with class labels
- `cifar100_class_counts.csv` — per-class training sample counts
- `cifar100_summary.json` — dataset summary statistics

Run the tests:

```powershell
.venv\Scripts\python -m pytest tests/test_cifar100.py -v
```
