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
Project scaffolding only. The folder structure and package layout have been
created; no data loading, model, federated learning, attack, attribution,
or repair logic has been implemented yet.
