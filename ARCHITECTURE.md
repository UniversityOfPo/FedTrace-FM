# FedTrace-FM — Software Architecture

> Describes the software architecture of FedTrace-FM as it exists today,
> plus the intended shape of flows that are not yet built (federated,
> attribution, foundation-model integration). No UML — every flow below is
> described as plain, ordered Markdown.

---

## Folder Structure

```
FedTrace-FM/
├── .venv/                  Local virtual environment (not part of the architecture; never committed)
├── configs/                YAML experiment configurations
├── data/                   CIFAR-100 downloads (gitignored)
├── notebooks/              Reserved for future exploratory notebooks (empty)
├── src/
│   ├── data/               Dataset loading, splitting, and inspection
│   ├── models/             Model architectures
│   ├── training/           Reusable, experiment-agnostic training/eval engine
│   ├── evaluation/         Reusable, experiment-agnostic metrics
│   ├── utils/              Configuration, seeding, checkpointing utilities
│   ├── federated/          Reserved for federated learning (empty)
│   ├── attacks/            Reserved for attack simulation (empty)
│   ├── attribution/        Reserved for FedTrace attribution logic (empty)
│   ├── repair/              Reserved for model repair logic (empty)
│   └── train_centralized.py  Centralized training entry point
├── experiments/            Reserved for future experiment orchestration scripts (empty)
├── results/                Generated outputs (gitignored except .gitkeep)
├── checkpoints/            Saved model checkpoints (gitignored except .gitkeep)
├── logs/                   Reserved (empty; per-run logs currently live under results/.../run.log)
├── tests/                  pytest test suite
├── requirements.txt
├── README.md
├── .gitignore
└── main.py                 Placeholder entry point
```

## Module Responsibilities

### `src/data/cifar100.py`
Owns everything about how CIFAR-100 is represented as PyTorch datasets:
normalization constants, train/eval transform pipelines, dataset
construction with automatic download (`build_datasets`), DataLoader
construction (`build_dataloaders`), class-name lookup (`get_class_names`),
summary statistics (`summarize_dataset`), and the stratified,
reproducible train/validation split (`split_train_validation`). Has no
knowledge of any model, training loop, or experiment configuration —
it only knows about data.

### `src/data/inspect_dataset.py`
A CLI script that uses `src/data/cifar100.py` and `src/utils/config.py` /
`src/utils/seed.py` to verify dataset integrity and produce reviewable
artifacts (sample grid image, per-class count CSV, summary JSON). Not
imported by any other module — it is a standalone diagnostic tool.

### `src/models/resnet18_cifar.py`
Owns the model architecture: a CIFAR-compatible ResNet-18 (`BasicBlock`,
`ResNetCifar`, the `resnet18_cifar()` factory) and a parameter-counting
utility (`count_parameters`). Has no knowledge of data, training loops, or
configuration — it only knows how to turn an image tensor into logits.

### `src/training/engine.py`
Owns the reusable mechanics of training and evaluating any model on any
DataLoader: device resolution, optimizer construction, scheduler
construction (with a clearly documented cosine/warmup/none policy), AMP
gradient-scaler construction, `train_one_epoch`, and `evaluate`. Imports
`topk_accuracy` from `src/evaluation/metrics.py` but otherwise has **no**
dependency on any specific experiment's configuration object, output
directory, checkpoint format, or dataset. This is deliberate: it is the
one module every future training loop (centralized today, federated
client training later) is expected to reuse unchanged.

### `src/evaluation/metrics.py`
Owns classification metrics: `topk_accuracy` (cheap, tensor-only, used
every batch), and end-of-run metrics computed on full prediction arrays
(confusion matrix, per-class accuracy, macro precision/recall/F1, balanced
accuracy). Depends only on `torch`, `numpy`, and `scikit-learn` — it has no
dependency on the training engine or any entry-point script, so it can be
reused by federated, attack, and attribution evaluation later without
pulling in `src/train_centralized.py`.

### `src/utils/config.py`
Owns typed, validated configuration loading. `load_yaml` is the generic
YAML-to-dict loader; `DataConfig`/`load_data_config` and
`TrainingConfig`/`load_training_config` are schema-specific typed wrappers
built on top of it, each raising `ConfigError` with a specific, actionable
message when a field is missing or out of range. Every entry-point script
loads its configuration through this module rather than parsing YAML
itself.

### `src/utils/seed.py`
Owns reproducibility: `set_seed()` seeds Python `random`, NumPy, and
PyTorch (CPU + all CUDA devices), and optionally enables full-determinism
mode (with a documented performance trade-off). Called once, at the very
start of every entry-point script, before any data loading or model
construction.

### `src/utils/checkpoint.py`
Owns checkpoint construction, atomic saving, loading, and state
restoration (`build_checkpoint`, `save_checkpoint`, `load_checkpoint`,
`restore_training_state`). Has no knowledge of any specific model
architecture or experiment configuration beyond what it is explicitly
handed — it operates on plain `state_dict()`s and a plain config
dictionary.

### `src/train_centralized.py`
The centralized training **entry point**. This is the one place in the
codebase that is allowed to know about all of the above modules at once:
it loads configuration, seeds RNGs, resolves the device, builds data
splits and loaders, builds the model/optimizer/scheduler/scaler, runs the
training loop (via `src/training/engine.py`), computes final metrics (via
`src/evaluation/metrics.py`), manages checkpoints (via
`src/utils/checkpoint.py`), and writes every reporting artifact. No other
module imports from this file — it is a leaf in the dependency graph, not
a library.

### Reserved packages: `src/federated/`, `src/attacks/`, `src/attribution/`, `src/repair/`
Currently contain only `__init__.py`. Their intended responsibilities are
described in the **Future Flows** sections below and in
`PROJECT_ROADMAP.md` (Phases 3–5).

## Data Flow

1. A YAML config under `configs/` specifies `dataset_config` (pointing at
   `configs/cifar100_data.yaml`), `data_root`, `download`, and split
   parameters (`validation_fraction`, `seed`).
2. `src/utils/config.py::load_data_config` loads and validates the data
   config into a `DataConfig`.
3. `src/data/cifar100.py::build_datasets` constructs the full CIFAR-100
   train dataset (train transforms) and test dataset (eval transforms),
   downloading into `data_root` if not already present.
4. `src/data/cifar100.py::split_train_validation` independently constructs
   two more `CIFAR100(train=True)` instances (one with train transforms,
   one with eval transforms) over the same on-disk data, computes a single
   stratified index split keyed on the configured seed, and wraps each
   instance in a `torch.utils.data.Subset` using those indices — producing
   a train subset (augmented) and a validation subset (clean), with no
   shared-object mutation.
5. `src/data/cifar100.py::build_dataloaders` wraps the train subset and
   validation subset in DataLoaders (shuffled train, non-shuffled
   validation); a separate plain `DataLoader` wraps the untouched test
   dataset.
6. Class names are read from the untouched test dataset
   (`get_class_names`), since it is guaranteed never to be subset or
   mutated.

## Training Flow

1. `src/train_centralized.py::main` loads and validates the training
   config (`TrainingConfig`), then calls `set_seed()`.
2. Device is resolved (`resolve_device`); if CUDA is selected, GPU name and
   VRAM are logged.
3. The Data Flow above runs to produce `train_loader`, `val_loader`, and
   `test_loader`.
4. The model is built (`resnet18_cifar`), moved to the resolved device, and
   its parameter counts are logged and printed.
5. Loss (`CrossEntropyLoss` with configured label smoothing), optimizer
   (`build_optimizer`: SGD or AdamW), scheduler (`build_scheduler`: cosine
   with optional warmup, or none), and AMP scaler (`build_grad_scaler`,
   automatically disabled on CPU) are constructed.
6. If `resume_from` is set, a prior checkpoint is loaded and training state
   (model/optimizer/scheduler/scaler) is restored; the starting epoch and
   best-validation-accuracy-so-far are recovered from the checkpoint.
7. For each epoch from the starting epoch to the configured epoch count:
   - `src/training/engine.py::train_one_epoch` runs one pass over
     `train_loader` (forward, loss, optional AMP scaling, optional gradient
     clipping, optimizer step), returning aggregate loss/top-1/top-5/
     elapsed time/learning rate.
   - `src/training/engine.py::evaluate` runs one pass over `val_loader`
     with gradients disabled, returning the same aggregate metrics (without
     retaining per-sample predictions — that mode is reserved for the final
     test evaluation).
   - The scheduler steps once (if present).
   - The epoch's metrics are appended to an in-memory history list.
   - If validation top-1 improved, the best-accuracy/best-epoch trackers
     update.
   - Checkpoints are saved per the **Checkpoint Flow** below.

## Evaluation Flow

Evaluation happens in two structurally different ways in the current
architecture:

1. **Per-epoch validation evaluation** (inside the Training Flow above):
   cheap, aggregate-only (`evaluate(..., return_predictions=False)`), run
   every epoch on `val_loader`, used purely to drive checkpoint selection
   and the training curves.
2. **Final test evaluation** (after training completes): the best
   checkpoint is reloaded into the model, then `evaluate(...,
   return_predictions=True)` runs exactly once on `test_loader`, returning
   both aggregate metrics and the full concatenated `(targets,
   predictions)` arrays. Those arrays feed
   `src/evaluation/metrics.py::compute_confusion_matrix`,
   `per_class_accuracy`, `macro_precision_recall_f1`, and
   `compute_balanced_accuracy`, producing the confusion matrix plot,
   per-class accuracy CSV, and classification metrics JSON. This is the
   **only** point in the entire pipeline where the official CIFAR-100 test
   set is evaluated.

## Checkpoint Flow

1. After every epoch, if `save_last_checkpoint` is enabled,
   `src/utils/checkpoint.py::build_checkpoint` assembles a full-state
   payload (epoch index, model/optimizer/scheduler/scaler state dicts,
   best validation top-1 so far, the fully resolved training configuration
   as a plain dict, seed, model name, class count, PyTorch version, CUDA
   version), and `save_checkpoint` writes it to `last.pt` under the
   experiment's checkpoint directory.
2. If validation top-1 improved this epoch **and** `save_best_checkpoint`
   is enabled, the same payload (with the updated best accuracy) is also
   written to `best.pt`.
3. `save_checkpoint` writes atomically: the payload is written to a
   temporary file in the same directory, then moved into place with
   `os.replace`, so a crash mid-write can never leave a corrupted
   checkpoint at the destination path.
4. After the training loop ends, if `best.pt` exists,
   `src/utils/checkpoint.py::load_checkpoint` reloads it and
   `restore_training_state` loads its model weights back into the model
   (optimizer/scheduler/scaler are not needed at this point, since no
   further training happens) before the one-time final test evaluation.
5. If `resume_from` is set at the start of a run, the same
   `load_checkpoint`/`restore_training_state` pair restores model,
   optimizer, scheduler, and scaler state, and the starting epoch/best
   accuracy are recovered from the checkpoint's stored `epoch` and
   `best_val_top1` fields.

## Future Federated Flow

Not yet implemented (`src/federated/` is currently empty). Intended shape:

1. A federated configuration specifies the number of simulated clients,
   the partitioning strategy (IID or non-IID), and the number of federated
   rounds.
2. A partitioning utility splits the (already leakage-free) CIFAR-100
   training subset from `split_train_validation` across the configured
   number of clients, producing one `Subset`-like dataset per client.
3. For each federated round:
   - A subset of clients is selected to participate.
   - Each participating client runs local training using
     `src/training/engine.py::train_one_epoch` (unmodified) for a
     configured number of local epochs, starting from the current global
     model weights.
   - Each client's resulting weights (or weight delta) are sent to an
     aggregation server.
   - The server aggregates client updates (e.g., FedAvg-style weighted
     averaging) into a new global model.
4. The global model is evaluated each round using
   `src/training/engine.py::evaluate` on the same held-out validation set
   used by the centralized baseline, so federated and centralized results
   remain directly comparable.
5. The official test set is still evaluated exactly once, at the end of
   the full federated run, using the best global model by validation
   accuracy — mirroring the Checkpoint/Evaluation Flow discipline
   established in the centralized baseline.

## Future Attribution Flow

Not yet implemented (`src/attribution/` is currently empty). Intended
shape, once attacks (Phase 4) exist within the Future Federated Flow:

1. During federated training, per-client update information (e.g., weight
   deltas, gradient statistics, or other signals TBD by the FedTrace
   design) is captured for each round rather than discarded immediately
   after aggregation.
2. After training (or periodically during training), the FedTrace
   attribution algorithm consumes this captured per-client information,
   along with the observed anomaly in global-model behavior, and produces
   a ranked or scored attribution of which client(s) most likely
   contributed the anomalous behavior.
3. Attribution quality is evaluated against the ground-truth malicious
   client labels that the attack simulation (Phase 4) already knows
   (since attacks are injected by the simulation itself), using metrics
   such as precision/recall of malicious-client identification.
4. This flow is designed to depend only on captured per-client update
   information and the existing evaluation metrics module — it does not
   require changes to `src/training/engine.py` itself, preserving that
   module's reusability.

## Future Foundation Model Integration

Not yet implemented. Intended shape:

1. `src/models/` gains a new model factory (parallel to
   `resnet18_cifar()`) that wraps a pretrained ViT/DINOv2/CLIP backbone
   with LoRA adapters, exposing the same interface expectations as the
   current model (a `forward()` producing logits, and compatibility with
   `count_parameters()`).
2. Because `src/training/engine.py` was deliberately kept model-agnostic
   (it only calls `model(images)` and standard PyTorch optimizer/scheduler
   APIs), the same `train_one_epoch`/`evaluate` functions are expected to
   work unchanged with a LoRA-adapted foundation-model backbone.
3. Only the LoRA-adapted parameters are registered as trainable (frozen
   backbone otherwise), keeping memory usage compatible with the local
   ~8 GB VRAM budget — this is the specific reason LoRA fine-tuning is the
   planned approach rather than full fine-tuning (see `DECISIONS.md`
   entries related to hardware constraints once foundation-model decisions
   are recorded there).
4. The Future Federated Flow, Future Attribution Flow, and the repair
   mechanism (Phase 5/`src/repair/`) are all expected to operate on top of
   this new model factory without structural changes, since none of them
   depend on the CNN-specific details of `resnet18_cifar.py`.
