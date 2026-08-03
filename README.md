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
CIFAR-100 data inspection pipeline and a centralized CIFAR-100 ResNet-18
baseline (training engine, checkpointing, evaluation metrics) are
implemented and frozen as Baseline v1.0. A reusable federated client
abstraction (`FederatedClient`, Sprint 3.1) is implemented. No federated
Server, aggregation (FedAvg or otherwise), communication-round
orchestration, client data partitioning, attacks, attribution, or
foundation-model logic has been implemented yet.

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

## Centralized ResNet-18 Baseline Usage

Configuration lives in
[configs/cifar100_resnet18_baseline.yaml](configs/cifar100_resnet18_baseline.yaml)
and references the CIFAR-100 data config for `data_root`/`download`/`pin_memory`
(the training config's own `batch_size`/`num_workers` always take precedence).

Run the full configured baseline (10 epochs by default) from the project root:

```powershell
.venv\Scripts\python -m src.train_centralized --config configs/cifar100_resnet18_baseline.yaml
```

This trains a CIFAR-compatible ResNet-18 (3x3 stride-1 stem, no initial
max-pool, no pretrained weights), evaluates after every epoch, saves
`best.pt`/`last.pt` checkpoints under `checkpoints/cifar100_resnet18/`,
reloads the best checkpoint for a final test evaluation, and writes to
`results/centralized/cifar100_resnet18/`:
- `history.csv` — per-epoch loss/accuracy/LR/duration
- `summary.json` — run-level summary (parameters, best/final metrics, timing, checkpoint paths)
- `training_loss.png`, `top1_accuracy.png`, `learning_rate.png` — training curves
- `confusion_matrix.png`, `per_class_accuracy.csv`, `classification_metrics.json` — final test-set breakdown
- `run.log` — full run log

Run the model/training/checkpoint tests:

```powershell
.venv\Scripts\python -m pytest tests/test_resnet18_cifar.py tests/test_training_engine.py -v
```

## Federated Client Abstraction (Sprint 3.1)

`src/federated/client.py` provides `FederatedClient` and
`FederatedClientConfig`: a single participant's purely local
computation — its own dataset, dataloaders, model, optimizer, scheduler,
and local checkpoint — built on top of the same reusable
`src/training/engine.py` used by the centralized baseline, so no training
logic is duplicated. The client knows nothing about FedAvg, a federated
Server, communication rounds, attacks, or attribution; those are later
sprints. There is no standalone CLI for this module yet, since it is a
library building block, not a runnable pipeline on its own.

Run the client tests:

```powershell
.venv\Scripts\python -m pytest tests/test_client.py -v
```
