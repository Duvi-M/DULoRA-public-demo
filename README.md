# DULoRA Public Demo

[English](README.md) | [Русский](README.ru.md)

**DULoRA: Data-Dependent Utility-Based Rank Allocation for Parameter-Efficient Transformer Fine-Tuning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-demo%20pipeline-red)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20integration-orange)](https://huggingface.co/docs/peft/)
[![Tests](https://img.shields.io/badge/tests-public%20smoke%20suite-brightgreen)](#quick-start)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DULoRA studies whether LoRA adapter capacity can be allocated non-uniformly
across transformer modules while staying within a controlled trainable-parameter
budget. This repository is the **sanitized public engineering demo** for that
larger research project.

It is designed for recruiters, engineers, and researchers who want to inspect
the project shape quickly: the problem, architecture, public code quality,
safe results summary, limitations, and the boundary between this demo and the
private research implementation.

![Public DULoRA pipeline](assets/public_pipeline.png)

> This public demo intentionally does **not** contain the unpublished DULoRA
> research estimator, adaptive decision policy, internal configurations, private
> layer-level assignments, or complete benchmark artifacts.

## What This Demonstrates

| Area | Public demo content |
| --- | --- |
| Problem | Fixed-rank LoRA can allocate the same adapter capacity to every selected module, even when modules may not need identical capacity. |
| Idea | Separate the rank-allocation decision from training/evaluation infrastructure through a clean allocator contract. |
| Pipeline | Data -> utility-estimation boundary -> rank allocation -> LoRA rank pattern -> training/evaluation. |
| Code | Installable `src/` Python package with PyTorch data, model, trainer, evaluator, reproducibility helpers, tests, and scripts. |
| Demo allocator | Deterministic round-robin placeholder used only to keep the public workflow executable. |
| ML integrations | Optional Hugging Face Transformers and PEFT builder for standard LoRA-style experimentation. |
| Results shown | Safe, high-level summaries only; demo metrics are toy runtime diagnostics. |

## Repository Scope

This repository is a public-safe companion to a larger private thesis/research
implementation. The public version preserves the engineering skeleton and
explainability of the project while withholding unpublished method details.

| Component | Public demo | Private research project |
| --- | --- | --- |
| Allocator interface | Included | Included |
| Deterministic placeholder allocator | Included | Not the research method |
| PyTorch training/evaluation loop | Included | Used in broader experiments |
| Synthetic offline data path | Included | Used for tests/smoke checks |
| Transformers/PEFT integration point | Included | Used for research workflows |
| DULoRA utility estimator | Withheld | Private |
| Adaptive allocation policy | Withheld | Private |
| Internal configs and layer traces | Withheld | Private |
| Complete unpublished benchmark artifacts | Withheld | Private |

## How It Works

The public package keeps rank allocation as a replaceable software boundary:

```python
from dulora_demo.demo_allocator import RoundRobinDemoAllocator

allocator = RoundRobinDemoAllocator(min_rank=1, max_rank=3, rank_step=1)
allocation = allocator.allocate(
    layer_names=[
        "encoder.block_a",
        "encoder.block_b",
        "encoder.block_c",
        "encoder.block_d",
    ],
    total_budget=8,
)

print(allocation.rank_pattern)
print(allocation.is_research_algorithm)  # False
```

The included allocator does not inspect gradients, activations, losses,
parameters, or model internals. It exists only so the public training path can
run end to end.

![Example public rank pattern](assets/rank_pattern_example.png)

## Safe Results Summary

The full research project includes experiments on BERT-based text
classification, ViT-based image classification, and LoRA/PEFT workflows.
Publicly safe claims are intentionally narrow:

| Finding type | Public-safe statement |
| --- | --- |
| Model families studied privately | BERT-style text classification and ViT-style image classification workflows. |
| PEFT context | LoRA-style parameter-efficient fine-tuning with controlled adapter budgets. |
| Parameter budget | A reported budget-96 setting reduces trainable adapter parameters by 50% compared with fixed LoRA rank 8. |
| Accuracy | Preliminary and dataset-dependent; this demo does not claim universal improvement over standard LoRA. |
| Public demo metrics | Smoke-test diagnostics from synthetic data, not benchmark evidence. |

![Parameter budget comparison](assets/parameter_comparison.png)

See [docs/results.md](docs/results.md) for the full public-safe results
boundary and interpretation notes.

## Quick Start

From `DULoRA-public-demo/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo/run_demo.py
python examples/minimal_text_classification.py
```

The default demo runs offline. It uses synthetic token IDs, generated labels,
placeholder layer names, a tiny PyTorch classifier, and the non-research demo
allocator.

## Demo Configuration

The public config is intentionally small and generic:

```yaml
seed: 7

dataset:
  source: synthetic
  train_size: 32
  eval_size: 16
  sequence_length: 12
  vocab_size: 64
  num_labels: 2

training:
  epochs: 1
  batch_size: 8
  learning_rate: 0.01

demo_allocator:
  strategy: round_robin
  layer_names:
    - encoder.block_a
    - encoder.block_b
    - encoder.block_c
    - encoder.block_d
  min_rank: 1
  max_rank: 3
  rank_step: 1
  total_budget: 8
```

Run a different config with:

```bash
python scripts/demo/run_demo.py --config configs/public/demo.yaml
```

These settings are not private experiment settings and should not be read as
thesis hyperparameters.

## Project Structure

```text
DULoRA-public-demo/
├── assets/                  # Public-safe README visuals
├── configs/public/          # Small synthetic demo config
├── docs/                    # Architecture, methodology, results, reproducibility
├── examples/                # Minimal package usage
├── scripts/demo/            # Demo runners and asset generation
├── src/dulora_demo/         # Public Python package
└── tests/                   # Lightweight public test suite
```

Key files:

| File | Purpose |
| --- | --- |
| `src/dulora_demo/allocator_interface.py` | Public allocator contract. |
| `src/dulora_demo/demo_allocator.py` | Deterministic non-research allocator. |
| `src/dulora_demo/data.py` | Synthetic offline data and optional public dataset helper. |
| `src/dulora_demo/model.py` | Tiny PyTorch model and optional PEFT builder. |
| `src/dulora_demo/trainer.py` | Compact training/evaluation loop. |
| `scripts/demo/run_demo.py` | Config-driven public demo entry point. |
| `scripts/demo/generate_public_assets.py` | Recreates public-safe README figures. |

## Optional PEFT Integration

The optional builder shows how a supplied public/demo rank pattern can be passed
to a standard PEFT LoRA configuration. It does not derive a research assignment.

```python
from dulora_demo.model import build_peft_sequence_classifier

model = build_peft_sequence_classifier(
    model_name="distilbert-base-uncased",
    num_labels=2,
    rank_pattern=None,
    target_modules=("q_lin", "v_lin"),
)
```

Install optional dependencies with:

```bash
python -m pip install -e ".[ml]"
```

This path may download third-party artifacts from Hugging Face.

## Documentation

- [Architecture](docs/architecture.md)
- [Methodology overview](docs/methodology_overview.md)
- [Results boundary](docs/results.md)
- [Reproducibility](docs/reproducibility.md)

## My Contribution

This public demo presents the engineering and research-software work around
DULoRA:

- Defined a modular rank-allocation interface.
- Built a runnable PyTorch demonstration pipeline.
- Added deterministic synthetic fixtures for offline testing.
- Exposed optional PEFT/Transformers integration points.
- Organized documentation around public/private research boundaries.
- Preserved sensitive and unpublished research logic outside the public release.

## Limitations

- The included allocator is illustrative and deterministic; it is not DULoRA.
- Default data is synthetic and offline.
- Demo accuracy/F1 are smoke-test diagnostics, not research results.
- Complete thesis experiments, ablations, configs, and private layer-level
  artifacts are intentionally excluded.
- Public claims should be limited to the documented safe summaries until the
  thesis/manuscript process permits broader release.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Until an official
paper, thesis record, or preprint is available, cite this repository as a
software demonstration:

```text
Duvan Mendoza. DULoRA Public Demo, version 0.1.0, 2026.
https://github.com/Duvi-M/DULoRA-public-demo
```

## Author

Duvan Mendoza  
MSc Software Engineering and Big Data  
MEPhI - Moscow Engineering Physics Institute

Research interests: machine learning, NLP, transformer models,
parameter-efficient fine-tuning, LoRA rank allocation, and research software
engineering.

## License

The code and public-safe assets in this repository are released under the
[MIT License](LICENSE). The license applies only to material included in this
public repository.
