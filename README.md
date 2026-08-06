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
abstraction (`FederatedClient`, Sprint 3.1), a federated server
abstraction with the client-server update contract (`FederatedServer`,
`ClientUpdate`, `RoundRecord`, Sprint 3.2), a pluggable aggregation
interface with its first strategy (`AggregationStrategy`, `FedAvg`,
Sprint 3.3), a strategy-agnostic round orchestrator (`FederatedRunner`,
Sprint 3.4), and the first real IID federated CIFAR-100 experiment
(`src/experiments/run_federated_cifar100.py`, Sprint 4.1) are implemented.
An explicit, round-level learning-rate scheduling mechanism for
`FederatedClient` (Sprint 5.1), now wired into
`src/experiments/run_federated_cifar100.py`'s YAML config loading
(Sprint 5.2), is also implemented. No Non-IID/Dirichlet/shard partitioning,
attacks, attribution, provenance, repair, or foundation-model logic has
been implemented yet.

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

## Federated Server & Client-Server Contract (Sprint 3.2)

`src/federated/contracts.py` defines the two data shapes exchanged
between a client and the server: `ClientUpdate` (a client's reported
result for one round — model state, sample count, metrics; always
immutable, CPU-cloned tensors, no optimizer/scheduler state) and
`RoundRecord` (server-side bookkeeping of who was selected/received/
rejected for one round). `src/federated/server.py` provides
`FederatedServer`/`FederatedServerConfig`: it owns the global model,
registered-client metadata, and round state, with deterministic
seed-based client selection. The server validates and stores updates via
`receive_update` but **never aggregates them or mutates the global
model** — no FedAvg or other aggregation strategy exists yet, and the
server never trains a client itself.

Run the server and contract tests:

```powershell
.venv\Scripts\python -m pytest tests/test_server.py -v
```

## Pluggable Aggregation & FedAvg (Sprint 3.3)

`src/federated/aggregation/base.py` defines `AggregationStrategy` (the
plugin interface every aggregation strategy implements) and
`AggregationResult` (an immutable, validated result: aggregated state
dict, participating client ids, total weight, metadata). `fedavg.py`
implements the first strategy, `FedAvg`: sample-count-weighted averaging
of floating-point tensors. Non-floating tensors split into two policies:
most (e.g. an arbitrary integer/boolean buffer) are required to be
identical across clients and passed through unchanged rather than
numerically averaged; keys ending with `.num_batches_tracked` (Sprint
6.1) — BatchNorm's cumulative local-batch counter, expected to differ
across clients that process different numbers of local batches (e.g.
under Non-IID/Dirichlet partitioning) — instead aggregate as the maximum
value submitted across participating clients. Aggregation is pure —
`FedAvg.aggregate()` never mutates its inputs and never touches
`FederatedServer`; the caller decides whether to load a result via the
existing `FederatedServer.load_global_weights()` (no server changes were
needed for either the original strategy or this fix).

Run the aggregation tests:

```powershell
.venv\Scripts\python -m pytest tests/test_aggregation.py -v
```

## Federated Round Orchestrator (Sprint 3.4)

`src/federated/runner.py` provides `FederatedRunner`, which connects an
already-constructed `FederatedServer`, a mapping of already-constructed
`FederatedClient`s, and any `AggregationStrategy` into communication
rounds — it constructs neither the server nor the clients itself. Per
round: the server selects clients, each selected client trains locally
and submits a `ClientUpdate`, the injected strategy aggregates the
accepted updates, and the result is **explicitly** loaded into the
server's global model (a failed round — insufficient updates, a client
error under `fail_round_on_client_error=True`, or an aggregation
exception — never touches the global model). `FederatedRoundResult`
records the outcome; an optional evaluation callback and an optional
per-round callback can be supplied. Every client passed to the runner
must already have had `initialize_model()` called on it.

Run the runner tests:

```powershell
.venv\Scripts\python -m pytest tests/test_runner.py -v
```

## First IID Federated CIFAR-100 Experiment (Sprint 4.1)

`src/data/federated_partition.py` provides `partition_dataset_iid`, a
generic (not CIFAR-specific) seeded-shuffle partitioner for any map-style
dataset. `src/experiments/run_federated_cifar100.py` is the first runnable
federated experiment: it partitions the existing 45,000-image training
subset IID across `num_clients`, keeps the existing 5,000-image validation
subset centralized for per-round evaluation, and evaluates the final
global model exactly once on the untouched 10,000-image test set. It
introduces no new federated abstractions — it only wires together
`FederatedClient`, `FederatedServer`, `FedAvg`, and `FederatedRunner`.

**Client-state policy:** at the start of every round, `load_global_weights`
now also rebuilds the client's optimizer/scheduler/AMP scaler from
scratch (`FederatedClient.reset_training_state()`) — no SGD momentum or
scheduler state carries over between communication rounds. This is the
baseline FedAvg client-state policy for this project.

Run the smoke test (2 rounds, 5 clients, 2 selected/round):

```powershell
.venv\Scripts\python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_iid_smoke.yaml
```

Run the full experiment (10 rounds, 5 clients, all selected/round):

```powershell
.venv\Scripts\python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_iid.yaml
```

Outputs are written to `results/federated/cifar100_fedavg_iid[_smoke]/`
(`round_history.csv`, `summary.json`, `final_test_metrics.json`,
`client_partition_summary.csv`, `client_partition_indices.json`,
validation/training/duration plots, `confusion_matrix.png`,
`per_class_accuracy.csv`, `run.log`) and checkpoints to
`checkpoints/federated/cifar100_fedavg_iid[_smoke]/` (`runner_latest.pt`
plus its `_server.pt`/`_client_<id>.pt` companions, per
`FederatedRunner.save_runner_checkpoint`'s existing naming convention,
and `final_global_model.pt`).

Run the partition and experiment tests:

```powershell
.venv\Scripts\python -m pytest tests/test_federated_partition.py tests/test_federated_experiment.py -v
```

## Round-Level Learning-Rate Scheduling (Sprint 5.1)

`src/federated/client.py` adds an explicit, round-level learning-rate
policy to `FederatedClient`/`FederatedClientConfig`, independent of the
existing local `scheduler` field. The local `scheduler` is scoped to
`local_epochs` and rebuilt from scratch every round (Sprint 4.1's
reset-every-round client-state policy) — with `local_epochs=1`, it cannot
vary the learning rate across communication rounds at all. The new
`round_lr_schedule` (`"constant"` or `"cosine"`), `min_learning_rate`, and
`total_communication_rounds` config fields drive a deterministic,
closed-form calculation (`compute_round_learning_rate`) that needs no
scheduler state to persist across rounds, so a cosine schedule can
meaningfully anneal from `learning_rate` at round 0 down to
`min_learning_rate` at the final round.

`FederatedRunner.run_round()` passes its `round_id` into each selected
client's `load_global_weights(state_dict, round_id=round_id)`, which
applies the round's learning rate immediately after rebuilding the
optimizer — before that round's local training begins. Both
`round_id` parameters default to `None`, and `round_lr_schedule` defaults
to `"constant"`, so every existing call site and experiment config is
behaviorally unchanged.

`src/experiments/run_federated_cifar100.py` (Sprint 5.2) now reads the
optional `round_lr_schedule`/`min_learning_rate` YAML fields (defaulting to
`"constant"`/`0.0` when absent, so `EXP_002`-`EXP_004`'s configs remain
valid and behaviorally unchanged), validates them, and forwards them —
plus `total_communication_rounds`, always taken from the config's own
`communication_rounds` rather than a separate YAML key — into every
client's `FederatedClientConfig`.
`configs/cifar100_fedavg_iid_50rounds_cosine.yaml` (`EXP_005`) is the first
config to set `round_lr_schedule: cosine`; it has not been run yet.

Run the client and runner tests (includes the new round-level
learning-rate cases):

```powershell
.venv\Scripts\python -m pytest tests/test_client.py tests/test_runner.py -v
```
