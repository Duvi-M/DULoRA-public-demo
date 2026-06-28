# DULoRA Public Demo: Dynamic Utility-based LoRA Rank Allocation

### Public-safe research software demo for modular LoRA workflows

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft/)
[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Master's%20Thesis-purple)](#academic-context)

DULoRA is an ongoing master's thesis research project studying **non-uniform
LoRA rank organization for parameter-efficient fine-tuning**.

This repository is the **public demo edition**. It keeps the same professional
research-software presentation style as the private development repository, but
publishes only a safe, simplified implementation. The goal is to demonstrate the
engineering workflow around DULoRA without exposing unpublished research logic.

> **This repository demonstrates the software architecture and public workflow
> only. It does not contain the research-specific DULoRA scoring or adaptive
> allocation method.**

The demo is built with PyTorch and includes optional integration points for
Hugging Face Transformers and PEFT. It uses synthetic/offline data, a compact
toy classifier, a public allocator interface, and a deterministic demonstration
allocator.

---

## Table of Contents

- [Research Motivation](#research-motivation)
- [Research Objective](#research-objective)
- [Method Overview](#method-overview)
- [Algorithm](#algorithm)
- [Adaptive Rank Allocation Algorithm](#adaptive-rank-allocation-algorithm)
- [Pipeline Overview](#pipeline-overview)
- [Current Public Demo Results](#current-public-demo-results)
- [Generated Demo Outputs](#generated-demo-outputs)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run dashboard](#run-dashboard)
- [Running an Experiment](#running-an-experiment)
- [How to reproduce the public demo](#how-to-reproduce-the-public-demo)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Current Evaluation](#current-evaluation)
- [Example Research Questions](#example-research-questions)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Academic Context](#academic-context)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

---

## Research Motivation

Low-Rank Adaptation, or LoRA, is a parameter-efficient fine-tuning technique
that reduces the number of trainable parameters required to adapt large
pretrained models to downstream tasks.

In many LoRA workflows, the same rank value is applied to every selected target
module. This is simple and often effective, but it also raises a natural
research question: do all selected modules need the same adaptation capacity?

DULoRA explores that question from a research perspective. The private research
version investigates whether LoRA capacity can be organized more selectively
while staying within a controlled budget. This public demo keeps the surrounding
software design visible while intentionally withholding the unpublished decision
logic.

---

## Research Objective

The broader DULoRA project aims to develop and evaluate a reproducible workflow
for studying non-uniform LoRA rank organization in transformer fine-tuning.

This public repository focuses on the safe engineering subset:

- Provide a clean, installable Python package using a `src/` layout.
- Demonstrate a modular LoRA-style allocation interface.
- Run an offline PyTorch training and evaluation pipeline.
- Expose optional Transformers/PEFT integration points.
- Use small synthetic fixtures instead of private experiments or datasets.
- Include tests for software contracts, metrics, and end-to-end execution.
- Document the public workflow clearly without revealing unpublished method
  details.

The research-specific estimator, adaptive decision policy, internal
configuration values, and final experimental artifacts are not included.

---

## Method Overview

The private research workflow and this public demo share a similar high-level
shape, but not the same internal algorithm.

### 1. Baseline LoRA context

The research context begins from the common LoRA setting: selected model modules
receive a low-rank adapter configuration. The public demo describes this context
conceptually but does not publish final private hyperparameters or benchmark
settings.

### 2. Public allocator contract

The demo exposes a stable Python interface for an allocator-like component. The
interface allows the rest of the training pipeline to request a module-to-rank
assignment without knowing how that assignment is produced.

This is useful for software architecture because allocator implementations can
be swapped without rewriting the data, model, trainer, or evaluator modules.

### 3. Demonstration allocator

The public implementation uses a deterministic round-robin style allocator
created only for this repository. It is intentionally simple:

- It does not inspect gradients.
- It does not inspect activations.
- It does not use losses or training dynamics.
- It does not reproduce the private DULoRA policy.
- It exists only to make the public pipeline executable.

---

## Algorithm

The public repository does **not** publish the research algorithm.

Instead, this section defines the release boundary. The private version contains
the unpublished DULoRA decision logic; the public version contains only a
software-compatible demonstration path.

This repository is intentionally explicit about what is and is not being
released.

| Component | Public Demo | Internal Research Version |
| --- | --- | --- |
| Allocator software interface | Included | Included internally |
| Deterministic demo allocator | Included for demonstration | Not used as the research method |
| PyTorch training loop | Included | Used in broader experimentation |
| Evaluation utilities | Included | Used in broader experimentation |
| Transformers/PEFT integration point | Included as optional code | Used internally |
| Research-specific estimator | Withheld | Maintained privately |
| Research adaptive decision policy | Withheld | Maintained privately |
| Layer-level research assignments | Not published | Used internally |
| Final experimental configurations | Not published | Used internally |
| Thesis figures and benchmark tables | Not published here | Used internally |

This boundary is deliberate while the thesis and associated manuscript are in
preparation.

---

## Adaptive Rank Allocation Algorithm

The adaptive research method is intentionally withheld from this public demo.
To keep the project runnable, the repository provides a simple demonstration
allocator behind the same kind of interface a research allocator would use.

The public allocator API is designed as a software contract, not as a disclosure
of the research method.

```python
from dulora_demo import AllocationRequest, RoundRobinDemoAllocator

request = AllocationRequest(
    module_names=("encoder.block_a", "encoder.block_b", "encoder.block_c"),
    min_rank=2,
    max_rank=6,
    total_rank_budget=8,
)

allocator = RoundRobinDemoAllocator()
result = allocator.allocate(request)
print(result.rank_pattern)
```

The returned assignment is deterministic and synthetic. It is safe for examples,
tests, and portfolio review, but it should not be interpreted as the DULoRA
research algorithm.

---

## Pipeline Overview

The public demo pipeline can be summarized as follows:

```text
Synthetic text-like samples
     ↓
Toy PyTorch classifier
     ↓
Public allocator interface
     ↓
Deterministic demo allocator
     ↓
Short training loop
     ↓
Evaluation
     ↓
Demo-only metrics and tests
```

The private research pipeline contains additional unpublished components that
are intentionally not part of this release.

---

## Current Public Demo Results

The default run is an offline smoke test, not an academic experiment. It uses
generated token IDs, generated labels, placeholder module names, and the public
demo allocator.

| Demo check | Public result |
| --- | --- |
| Allocator implementation | `RoundRobinDemoAllocator` |
| Input data | Synthetic/offline |
| Placeholder modules | 4 |
| Total demonstration rank | 8 |
| Assigned demonstration ranks | 2 for each placeholder module |
| Research algorithm included | No |
| End-to-end training | Completes successfully |
| Test suite | 8 tests passing |

Accuracy, F1, and loss printed by the scripts are **toy runtime diagnostics**.
They are not benchmark results and must not be cited as DULoRA research
performance.

---

## Generated Demo Outputs

The public scripts print metrics to the terminal and are designed to run without
creating persistent experiment outputs. If you extend this repository, keep
generated artifacts outside version control.

Recommended local-only locations include:

```text
outputs/
checkpoints/
logs/
wandb/
mlruns/
models/
```

These paths are ignored by `.gitignore`.

---

## Project Structure

```text
DULoRA-public-demo/
│
├── assets/
│   └── .gitkeep
│
├── configs/
│   └── public/
│       └── demo.yaml
│
├── docs/
│   ├── methodology_overview.md
│   └── reproducibility.md
│
├── examples/
│   └── minimal_text_classification.py
│
├── scripts/
│   └── demo/
│       └── run_demo.py
│
├── src/
│   └── dulora_demo/
│       ├── __init__.py
│       ├── allocator_interface.py
│       ├── data.py
│       ├── demo_allocator.py
│       ├── evaluator.py
│       ├── model.py
│       ├── reproducibility.py
│       └── trainer.py
│
├── tests/
│   ├── test_allocator_contract.py
│   ├── test_metrics.py
│   └── test_pipeline.py
│
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Main Components

| File | Purpose |
| --- | --- |
| `src/dulora_demo/allocator_interface.py` | Defines the public allocator contract. |
| `src/dulora_demo/demo_allocator.py` | Provides the deterministic non-research demo allocator. |
| `src/dulora_demo/data.py` | Creates synthetic offline datasets and optional dataset helpers. |
| `src/dulora_demo/model.py` | Builds the toy classifier and optional PEFT sequence classifier. |
| `src/dulora_demo/trainer.py` | Runs a compact PyTorch training/evaluation loop. |
| `src/dulora_demo/evaluator.py` | Computes accuracy and binary F1. |
| `src/dulora_demo/reproducibility.py` | Applies seed and deterministic runtime settings. |
| `configs/public/demo.yaml` | Stores small public demo settings. |
| `scripts/demo/run_demo.py` | Runs the configuration-driven public demo. |
| `examples/minimal_text_classification.py` | Shows minimal package usage. |

---

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development dependencies for testing:

```bash
python -m pip install -e ".[dev]"
```

Install optional ML integrations:

```bash
python -m pip install -e ".[ml]"
```

Or install all public demo features at once:

```bash
python -m pip install -r requirements.txt
```

---

## Run dashboard

The original private development repository may include dashboard-style
visualization around internal experiment artifacts. This public demo does not
ship a dashboard because it does not publish those artifacts.

A future public dashboard could visualize only synthetic demo outputs.

---

## Running an Experiment

Run the configuration-driven demo:

```bash
python scripts/demo/run_demo.py
```

Run the minimal package example:

```bash
python examples/minimal_text_classification.py
```

Both commands use synthetic inputs and do not download datasets or pretrained
models.

---

## How to reproduce the public demo

From the repository root:

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo/run_demo.py
python examples/minimal_text_classification.py
python -m compileall -q src examples scripts tests
```

The expected result is a passing test suite and terminal output showing the demo
allocator strategy, a synthetic module assignment, and toy evaluation metrics.

---

## Configuration

The public configuration is intentionally small and generic:

```yaml
seed: 42

dataset:
  train_size: 24
  eval_size: 8
  sequence_length: 12
  vocab_size: 128

model:
  hidden_size: 32
  num_labels: 2

training:
  epochs: 2
  batch_size: 8
  learning_rate: 0.001

demo_allocator:
  min_rank: 2
  max_rank: 6
  total_rank_budget: 8
  module_names:
    - encoder.block_a
    - encoder.block_b
    - encoder.block_c
    - encoder.block_d
```

### Configuration Fields

| Field | Description |
| --- | --- |
| `seed` | Random seed used by the public demo. |
| `dataset.train_size` | Number of synthetic training samples. |
| `dataset.eval_size` | Number of synthetic evaluation samples. |
| `dataset.sequence_length` | Synthetic token sequence length. |
| `dataset.vocab_size` | Synthetic vocabulary size for the toy classifier. |
| `model.hidden_size` | Hidden dimension of the compact PyTorch model. |
| `model.num_labels` | Number of output labels. |
| `training.epochs` | Number of toy training epochs. |
| `training.batch_size` | Batch size for the public demo. |
| `training.learning_rate` | Optimizer learning rate for the toy run. |
| `demo_allocator.min_rank` | Minimum demonstration rank allowed by the public allocator contract. |
| `demo_allocator.max_rank` | Maximum demonstration rank allowed by the public allocator contract. |
| `demo_allocator.total_rank_budget` | Small synthetic budget for the demo assignment. |
| `demo_allocator.module_names` | Placeholder module names used by the public example. |

These values are not private experiment settings and are not intended to
reproduce thesis results.

---

## Optional Transformers and PEFT Integration

The optional builder demonstrates standard library integration with a supplied
public or synthetic assignment. It does not derive a research assignment.

```python
from dulora_demo.model import build_peft_sequence_classifier

model = build_peft_sequence_classifier(
    model_name="distilbert-base-uncased",
    num_labels=2,
    rank_pattern=None,
    target_modules=("q_lin", "v_lin"),
)
```

This example requires:

```bash
python -m pip install -e ".[ml]"
```

Calling the builder may download third-party artifacts from Hugging Face.

---

## Testing

Run the public test suite:

```bash
pytest -q
```

The tests cover:

- Allocator contract behavior.
- Deterministic demo assignment.
- Seed control.
- Accuracy and binary F1.
- End-to-end offline training.

The tests do not encode, approximate, or validate the private research method.

---

## Reproducibility

The demo entry points set the global seed before model creation and training.
The helper covers Python, PyTorch, and CUDA seeding when CUDA is available. It
also requests best-effort deterministic PyTorch behavior.

Exact reproducibility may still depend on hardware, backend behavior, operating
system, CUDA/cuDNN settings, and library versions.

See [docs/reproducibility.md](docs/reproducibility.md) for additional details.

---

## Current Evaluation

The public demo reports:

- Training loss.
- Accuracy.
- Binary F1.
- Demo allocator strategy.
- Synthetic assignment summary.

These values are printed for smoke testing only. They are not stored as
published results and are not evidence for or against the private research
hypothesis.

---

## Example Research Questions

The full research project is designed around questions such as:

- Can LoRA capacity be organized more selectively than a uniform setup?
- How should a fair comparison control for trainable parameter budget?
- How stable are non-uniform assignments across seeds, tasks, and model
  families?
- Which evaluation protocols are most appropriate for parameter-efficient
  adaptation research?
- How can research software expose enough structure for reproducibility without
  prematurely releasing unpublished intellectual property?

This public repository supports only the software-engineering side of those
questions.

---

## Current Limitations

The public demo has deliberate limitations:

- It uses synthetic/offline data.
- It uses a compact toy classifier by default.
- It does not publish thesis experiments or benchmark results.
- It does not include private method details.
- It does not provide final ablations, internal configs, or layer-level
  research artifacts.
- Optional Transformers/PEFT integration is provided as a safe extension point,
  not as a full reproduction package.

These limitations are part of the public-release boundary.

---

## Future Work

Possible public extensions include:

- Adding more toy tasks that remain offline and safe.
- Publishing additional documentation around experiment hygiene.
- Providing more tests for package installation and CLI behavior.
- Adding a small dashboard that visualizes synthetic demo outputs only.
- Releasing a formal reproduction package after the thesis/manuscript process
  allows it.
- Publishing research-specific components later under an appropriate archival
  release.

---

## Academic Context

This repository was developed as a public-safe companion to a master's thesis
project on parameter-efficient fine-tuning of transformer models.

The public release is intended for portfolio review, research-software
inspection, and reproducibility practice. It is not the complete DULoRA research
implementation.

---

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

Until an official paper, thesis record, or preprint is available, cite this
repository as a software demonstration:

```text
Duvan Mendoza. DULoRA Public Demo, version 0.1.0, 2026.
https://github.com/Duvi-M/DULoRA-public-demo
```

---

## Author

Duvan Mendoza  
MSc Software Engineering and Big Data  
MEPhI - Moscow Engineering Physics Institute

Research interests:

- Machine Learning
- Natural Language Processing
- Transformer Models
- Parameter-Efficient Fine-Tuning
- LoRA Rank Allocation
- Research Software Engineering

---

## License

The code and public-safe assets in this repository are released under the
[MIT License](LICENSE).

The license applies only to material included in this public repository.
Research components that are not distributed here are outside the scope of this
license.
