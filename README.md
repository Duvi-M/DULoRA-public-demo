# DULoRA Public Demo

DULoRA is an ongoing research project investigating utility-guided rank
allocation for parameter-efficient fine-tuning with Low-Rank Adaptation
(LoRA).

Standard LoRA commonly applies one rank to every selected module. DULoRA
studies the broader research question of whether adaptation capacity can be
distributed more selectively while retaining a controlled parameter budget.

This repository is a **sanitized, functional software demonstration**. It
shows the engineering architecture, configuration flow, PyTorch training,
evaluation, and optional Hugging Face Transformers/PEFT integration without
publishing the research-specific scoring or allocation method.

## Public release scope

Included:

- A small offline PyTorch text-classification pipeline.
- A stable rank-allocator interface.
- A deterministic round-robin allocator used only for demonstration.
- Optional Transformers and PEFT model construction.
- Public YAML configuration.
- Contract, pipeline, and metric tests.
- Reproducibility documentation.

Temporarily omitted while the thesis and manuscript are in preparation:

- Research-specific layer-utility scoring.
- The adaptive allocation policy used in internal experiments.
- Internal hyperparameter selections and ablations.
- Layer-level scores, rank patterns, and unpublished experiment outputs.

The `RoundRobinDemoAllocator` is intentionally simple and is **not** the
DULoRA research algorithm.

## High-level pipeline

```text
Public or synthetic dataset
        |
        v
Tokenization / batching
        |
        v
PyTorch or Transformers model
        |
        v
Public allocator interface
        |
        v
Demonstration rank pattern
        |
        v
Training and evaluation
```

The optional PEFT builder accepts an already prepared rank pattern. The public
demo does not infer that pattern from gradients, activations, losses, or other
model-internal signals.

## Repository structure

```text
src/dulora_demo/       Public Python package
configs/public/        Small demonstration configuration
examples/              Minimal executable example
scripts/demo/          Configuration-driven demo entry point
tests/                 Public behavioral tests
docs/                  Methodology and reproducibility notes
assets/                Public-safe visual assets
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base installation contains the dependencies required by the offline demo.
For tests, install the development extra:

```bash
python -m pip install -e ".[dev]"
```

For the optional Hugging Face Transformers/PEFT integration:

```bash
python -m pip install -e ".[ml]"
```

To install all public features at once:

```bash
python -m pip install -r requirements.txt
```

## Run the offline demo

The default demo uses generated token IDs and does not download a dataset or
pretrained model.

```bash
python scripts/demo/run_demo.py
```

Or run the minimal example:

```bash
python examples/minimal_text_classification.py
```

## Run tests

```bash
pytest -q
```

The tests verify interface behavior, budget constraints, deterministic demo
behavior, metric correctness, and an end-to-end offline training pass. They do
not encode or validate the private research algorithm.

## Transformers and PEFT integration

`dulora_demo.model.build_peft_sequence_classifier` demonstrates how a
pre-existing public or synthetic rank pattern can be passed to PEFT. Calling
it may download a pretrained model from Hugging Face and requires the `ml`
optional dependencies.

```python
from dulora_demo.model import build_peft_sequence_classifier

model = build_peft_sequence_classifier(
    model_name="distilbert-base-uncased",
    num_labels=2,
    rank_pattern=None,
    target_modules=("q_lin", "v_lin"),
)
```

## Research status

DULoRA is under active academic development. Public interfaces and
documentation may evolve before a thesis, preprint, or manuscript release.
Claims about the research method should not be inferred from the deliberately
non-adaptive round-robin demonstration included here.

The default example fixes Python and PyTorch random seeds and requests
best-effort deterministic PyTorch behavior. It verifies software execution;
it does not reproduce or approximate unpublished research experiments.

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for the public demo's
scope, deterministic settings, and limitations.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). Until an
official paper or preprint is available, cite this software repository as a
research software demonstration.

## License

The public demonstration code is released under the MIT License. The license
applies only to material present in this public repository; omitted research
materials are not part of this distribution.
