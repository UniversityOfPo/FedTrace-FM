# FedTrace-FM — Engineering Decisions

> Records every significant engineering decision made on this project: what
> was decided, why, what alternatives were considered, why they were
> rejected, and the expected long-term benefit. This file is meant to let a
> future contributor (human or AI) understand *why* the codebase looks the
> way it does, not just what it currently contains.

---

## Why Python

**Decision:** Use Python 3.12 as the sole implementation language for the
entire project.

**Reason:** Python is the de facto standard for deep learning research,
with first-class support in PyTorch, torchvision, scikit-learn, and the
broader federated-learning research ecosystem this project will eventually
draw on (attack simulation libraries, foundation-model tooling, etc.).
Using the same language across the data pipeline, model, training engine,
and (later) federated/attack/attribution/repair modules avoids any
cross-language interop overhead.

**Alternative Considered:** A compiled language (C++/Rust) for the
performance-critical training loop, with Python only for orchestration.

**Why Rejected:** The training loop's actual bottleneck on a single 8 GB
laptop GPU is GPU kernel execution (already implemented in optimized C++/
CUDA inside PyTorch itself), not Python-level orchestration overhead. A
mixed-language architecture would add substantial engineering cost
(binding layers, build complexity) for negligible performance gain at this
project's scale, while significantly slowing down the iterative,
exploratory research workflow this project needs.

**Long-Term Benefit:** A single-language codebase is easier for a research
collaborator (or a future AI assistant) to read, modify, and extend across
every planned phase — from the CIFAR-100 baseline through federated
learning, attacks, attribution, repair, and foundation-model integration.

---

## Why PyTorch

**Decision:** Use PyTorch (2.11.0, CUDA 12.8 build) as the deep learning
framework.

**Reason:** PyTorch has native, actively maintained support for automatic
mixed precision (`torch.amp`), a clean eager-mode API well suited to the
iterative research style this project needs, first-class GPU support
matching the local RTX 5060 Laptop GPU, and it is the dominant framework
in current federated learning and foundation-model research literature —
meaning LoRA-adapted ViT/DINOv2/CLIP integration (Phase 6) will have
abundant reference implementations to draw on.

**Alternative Considered:** TensorFlow/Keras.

**Why Rejected:** TensorFlow's more static graph-construction style (even
with eager mode) tends to add friction for the kind of research code this
project needs — custom training loops, per-client federated logic, and
experimental attribution mechanisms that don't fit neatly into
high-level `fit()`-style APIs. PyTorch's lower-level control over the
training loop maps more directly onto what `src/training/engine.py` needs
to do.

**Long-Term Benefit:** Direct compatibility with the broader federated
learning and LoRA/foundation-model ecosystem planned for later phases,
without needing a framework migration partway through the project.

---

## Why YAML

**Decision:** Use YAML files under `configs/` as the sole configuration
format, loaded into typed dataclasses (`DataConfig`, `TrainingConfig`) via
`src/utils/config.py`.

**Reason:** YAML is human-readable and diffable in version control, easy
to author and review by hand, and has a mature, simple Python parser
(`pyyaml`) already in `requirements.txt`. Wrapping parsed YAML in typed
dataclasses with explicit validation (`ConfigError` on missing/invalid
fields) gives the readability of YAML with the safety of typed, validated
configuration — catching misconfiguration before any GPU time is spent.

**Alternative Considered:** Plain Python config modules (e.g.,
`config.py` files defining variables directly) or JSON.

**Why Rejected:** Python config modules risk arbitrary code execution and
make diffs noisier (Python syntax vs. YAML's plain key-value structure);
JSON does not support comments, which reduces the self-documentation value
of a config file that future contributors need to understand at a glance.

**Long-Term Benefit:** Every experiment (centralized baseline, and later
every federated/attack/attribution configuration) is fully specified by a
single reviewable, versionable YAML file, making experiments reproducible
and comparable across the entire multi-paper research arc this project is
planned to support.

---

## Why CIFAR-100

**Decision:** Use CIFAR-100 as the first benchmark dataset.

**Reason:** CIFAR-100 is small enough (32x32 images, 60,000 total images)
to train quickly on a single 8 GB laptop GPU, while still being
substantially harder than CIFAR-10 (100 classes, only 500 training images
per class), making it a meaningful testbed for studying attribution and
repair under realistic classification difficulty. It is also a standard,
widely used benchmark in both centralized and federated learning
literature, which makes results directly comparable to prior work when
this project's findings are eventually written up.

**Alternative Considered:** ImageNet (or a large ImageNet subset).

**Why Rejected:** ImageNet-scale training is impractical within the local
~8 GB VRAM / single-GPU constraint for the number of experimental
configurations this project needs to run (multiple attack types, multiple
attribution/repair variants, multiple federated configurations). CIFAR-100
allows many more experimental iterations within the same hardware and time
budget.

**Long-Term Benefit:** Fast iteration during methodology development
(federated simulation, attack simulation, attribution, repair) before that
methodology is carried over to the more expensive foundation-model
(Phase 6) and medical-imaging (Phase 7) settings, where mistakes are much
costlier to discover.

---

## Why ResNet-18

**Decision:** Use a CIFAR-compatible ResNet-18 (custom 3x3 stride-1 stem,
no initial max-pool) as the first model architecture.

**Reason:** ResNet-18 is small enough to train many times over on a single
8 GB laptop GPU, architecturally simple enough to reason about when
designing attribution/repair mechanisms, and a well-understood, widely
used baseline in both centralized and federated CIFAR literature. The
CIFAR-specific stem modification (3x3 stride-1 convolution, no max-pool,
replacing the standard 7x7 stride-2 + max-pool ImageNet stem) is necessary
because CIFAR-100 images are only 32x32 — the standard ImageNet stem would
downsample them to 8x8 before the first residual stage even runs,
discarding most spatial detail.

**Alternative Considered:** A deeper ResNet variant (ResNet-34/50) or a
Vision Transformer from the start.

**Why Rejected:** A deeper CNN or a ViT trained from scratch would cost
significantly more GPU time per experiment, without adding value to the
current research questions (which are about federated attribution/repair
methodology, not about squeezing out maximum CIFAR-100 accuracy). Vision
Transformers are explicitly deferred to Phase 6, where they are introduced
specifically as LoRA-adapted pretrained backbones (not trained from
scratch), which is a different and more appropriate use case for them
given the local hardware budget.

**Long-Term Benefit:** A fast, well-understood baseline lets the
federated/attack/attribution/repair methodology (Phases 3–5) be validated
quickly and cheaply before it is carried over to the more expensive
foundation-model setting (Phase 6), where any methodology bugs would be
far more expensive to discover and fix.

---

## Why Validation Split

**Decision:** Carve a stratified, reproducible validation subset out of
the CIFAR-100 training set (`validation_fraction`, default 0.1), and
restrict the official test set to a single evaluation at the very end of
training.

**Reason:** The initial centralized-baseline implementation evaluated
every epoch directly against the official CIFAR-100 test set and used that
same test-set accuracy to select the "best" checkpoint. This is test-set
leakage: the reported "final test accuracy" was not an unbiased estimate
of generalization, because the test set had already influenced which
checkpoint was selected. Introducing a validation split (stratified so
all 100 classes remain proportionally represented, and seeded so the split
is reproducible) restores the standard, methodologically correct
train/validation/test discipline expected of any credible research result.

**Alternative Considered:** K-fold cross-validation over the training set.

**Why Rejected:** K-fold cross-validation multiplies training cost by the
number of folds, which is not justified at this stage given the number of
experimental configurations this project needs to run across federated,
attack, attribution, and repair variants. A single stratified holdout
split is the standard, sufficient choice for this project's current needs,
and matches how the later federated/attack/attribution experiments will
also need a clean validation signal per experiment.

**Long-Term Benefit:** Every future experiment built on top of
`src/train_centralized.py` (and, later, federated training) inherits a
leakage-free evaluation methodology by construction, rather than requiring
each new module to remember to implement this correctly on its own.

---

## Why Git

**Decision:** Use Git for version control, with commits created only when
explicitly requested, new commits preferred over amending history, and
generated artifacts (`results/`, `checkpoints/`, `data/`, `logs/`) excluded
from tracking.

**Reason:** Git is the standard for reproducible research code, integrates
directly with the GitHub-based collaboration and eventual public-release
workflow this project anticipates ("Assume this repository will support
multiple future papers" and eventually be public), and its commit history
itself becomes part of the project's own reproducibility record (see
`FEDTRACE_FM_MASTER.md` → *Current Git History*).

**Alternative Considered:** No version control, or manual folder-based
versioning (e.g., timestamped copies of the codebase).

**Why Rejected:** Manual folder-based versioning does not scale to a
multi-phase, multi-paper project, provides no diffing, no history search,
and no safe way to collaborate or roll back a bad change — all of which
this project will need increasingly as it grows more complex through
Phases 3–9.

**Long-Term Benefit:** A clean, meaningful commit history that itself
documents the project's development (see `RESEARCH_DIARY.md` for the
narrative alongside it), and a straightforward path to eventual public
release.

---

## Why Modular Design

**Decision:** Separate the codebase into focused packages
(`src/data/`, `src/models/`, `src/training/`, `src/evaluation/`,
`src/utils/`), with reserved empty packages for future work
(`src/federated/`, `src/attacks/`, `src/attribution/`, `src/repair/`), and
keep the training engine and evaluation metrics free of any dependency on
a specific experiment's configuration or entry-point script.

**Reason:** This project is explicitly planned to grow through nine
phases (`PROJECT_ROADMAP.md`) reusing the same core mechanics — most
importantly, the same `train_one_epoch`/`evaluate` functions are meant to
run unchanged inside future federated clients. Modularity with clear
module boundaries is what makes that reuse possible without duplicating
or forking the training loop for every new phase.

**Alternative Considered:** A single monolithic training script per
experiment (a common pattern in fast-and-loose research code).

**Why Rejected:** Monolithic per-experiment scripts are faster to write
for a single one-off experiment, but this project is explicitly not a
one-off — it is meant to support multiple future papers. Duplicating
training logic across a centralized script, a federated script, and
attack/attribution experiments would multiply the maintenance burden and
risk subtle inconsistencies (e.g., a bug fixed in one copy of the training
loop but not another) exactly like the test-set-leakage bug this project
already had to fix once.

**Long-Term Benefit:** A bug fix or improvement to the training engine
(e.g., the AMP/GradScaler handling, or gradient clipping) automatically
benefits every module that reuses it, present and future, instead of
needing to be re-applied in multiple places.

---

## Why Smoke Test

**Decision:** Before committing to a full multi-epoch training run,
validate the entire pipeline end-to-end with a cheap, short run (a 1-epoch
"smoke test" configuration, e.g.
`configs/cifar100_resnet18_smoke.yaml`).

**Reason:** A full 10-epoch baseline run costs meaningfully more time than
a 1-epoch run, and if a bug exists anywhere in the pipeline (data loading,
model construction, checkpoint saving, final evaluation, output-file
writing), a full run would waste that time before surfacing the bug. A
smoke test confirms the entire pipeline — including newly introduced
changes like the validation split — executes correctly on real hardware
in under a minute, before a longer run is started.

**Alternative Considered:** Rely solely on unit tests (CPU-only, synthetic
data) as sufficient verification before a full run.

**Why Rejected:** Unit tests intentionally use tiny synthetic data and
never touch real CIFAR-100, real GPU execution, real checkpoint I/O to
disk, or the real output-artifact-writing code paths in
`src/train_centralized.py`. A smoke test is the only step that exercises
the real dataset, the real GPU, and the entire artifact-writing pipeline
together, which is exactly the combination unit tests are not designed to
cover.

**Long-Term Benefit:** A cheap, fast confidence check that can be re-run
before every future full-scale training run (federated, foundation-model,
or medical-dataset experiments alike), catching integration bugs early
without paying the cost of a full run.

---

## Why Unit Tests

**Decision:** Maintain a lightweight, CPU-friendly, fast pytest suite
(`tests/`) covering configuration loading, dataset/split properties, model
architecture properties, training-engine behavior, metrics correctness,
and checkpoint round-tripping — with no test requiring a GPU or a full
training run to pass.

**Reason:** Unit tests catch a different class of bug than a smoke test:
precise, deterministic properties that are easy to get subtly wrong and
easy to verify in isolation — e.g., "does the stratified split actually
have zero index overlap between train and validation," "does the CIFAR
stem actually use a 3x3 stride-1 convolution instead of the ImageNet
7x7/stride-2 one," "does one training step actually change model
parameters while one evaluation step does not." These are exactly the
kinds of properties a smoke test's aggregate accuracy numbers would never
reveal if they were wrong.

**Alternative Considered:** Rely solely on smoke tests and manual
inspection of full training runs.

**Why Rejected:** Manual inspection does not scale as the number of
modules grows (federated partitioning, attack implementations,
attribution logic all need their own correctness checks), and full
training runs are far too slow and noisy a signal to catch a precise bug
like an off-by-one in split-index computation or an incorrectly wired
evaluation loader (the exact class of bug the test-set-leakage issue was).

**Long-Term Benefit:** A fast, CPU-only test suite that can run before
every commit, on any machine, without needing GPU access or a downloaded
dataset re-fetch, catching regressions immediately as the codebase grows
through every future phase.

---

## Why Documentation First

**Decision:** Treat documentation (`FEDTRACE_FM_MASTER.md`,
`RESEARCH_DIARY.md`, `PROJECT_ROADMAP.md`, `ARCHITECTURE.md`,
`DECISIONS.md`) as first-class deliverables, maintained continuously
alongside code rather than written retroactively at the end of the
project.

**Reason:** This project is explicitly designed to span many work
sessions, multiple contributors (human and AI), and eventually multiple
published papers. Without a maintained master memory, diary, roadmap,
architecture description, and decision log, a future contributor (or a
future AI assistant with no memory of this conversation) would have to
reverse-engineer intent and history from code and commit messages alone —
which is lossy and error-prone, especially for *why* decisions were made,
not just *what* was implemented.

**Alternative Considered:** Rely on code comments and commit messages
alone, writing a paper-style writeup only at the end (Phase 8/9).

**Why Rejected:** Code comments and commit messages capture *what* changed
far better than *why* a design was chosen over its alternatives, and
end-of-project documentation risks losing details that were obvious in
the moment but forgotten months later — exactly the details (like why the
validation split was needed, or why `torch.amp` was chosen over
`torch.cuda.amp`) that make a research codebase trustworthy and
reproducible.

**Long-Term Benefit:** Any future session — including one run by a
different AI model with zero conversation history — can read
`FEDTRACE_FM_MASTER.md` and the supporting documents and resume work
correctly, without needing to re-derive context that already exists.

---

## Why Results Are Ignored in Git

**Decision:** Exclude the contents of `results/`, `checkpoints/`, `data/`,
and `logs/` from version control via `.gitignore` (keeping only
`.gitkeep` placeholders so the folder structure itself is preserved).

**Reason:** These directories contain large, regenerable binary and
generated-artifact content: downloaded dataset files, model checkpoints
(tens of megabytes each), and plots/CSVs/JSON summaries produced by
running the code against a specific config and seed. None of this is
source — it is entirely reproducible by running the corresponding config
through the corresponding script, given the same seed.

**Alternative Considered:** Commit results and checkpoints directly to the
git repository, so any commit's exact outputs are always available without
re-running anything.

**Why Rejected:** Committing large binary artifacts (dataset files,
`.pt` checkpoints) bloats repository size dramatically, slows down every
clone/fetch/checkout operation, and does not actually add reproducibility
value here, since the whole point of the typed-config + seeded pipeline
built in this project is that results are deterministically regenerable
from source. Git is also poorly suited to diffing or merging binary
checkpoint files.

**Long-Term Benefit:** A small, fast, purely-source repository that scales
cleanly across every future phase (federated results, attack results,
attribution results, foundation-model checkpoints, medical-dataset
results) without ever needing to prune or rewrite history to remove
accumulated binary bloat — while reproducibility is instead guaranteed by
the versioned configs, seeds, and code that produced each result.
