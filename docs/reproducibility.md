# Reproducibility

## Public demo scope

The default run is an offline software smoke test. It generates small token
tensors, trains a compact PyTorch classifier for one epoch, and evaluates it on
a generated holdout set.

It is designed to verify that the public pipeline works; it is not an
experiment supporting claims about the private DULoRA method.

## Deterministic inputs

`configs/public/demo.yaml` defines small, generic values. Synthetic data uses a
fixed seed and the demonstration allocator is deterministic for a fixed layer
order and budget.

Before model creation, the entry points seed Python and PyTorch. CUDA devices
are seeded when available. PyTorch deterministic algorithms are requested in
warning mode, and cuDNN benchmarking is disabled when that backend exists.

Floating-point training results may still differ slightly across:

- Operating systems.
- CPU, CUDA, or Apple Silicon backends.
- PyTorch versions.
- Low-level linear algebra implementations.

## Recommended environment

- Python 3.10+
- A fresh virtual environment
- An editable installation using `python -m pip install -e ".[dev]"`
- CPU execution for the default smoke test

## Verification commands

```bash
python scripts/demo/run_demo.py
pytest -q
```

## Optional online components

The Transformers/PEFT builder and Hugging Face dataset helper require network
access and may download third-party models or datasets. Their licenses and
terms remain the responsibility of their respective publishers.

No pretrained model, dataset, checkpoint, or experiment output is bundled in
this repository.

## Research reproducibility

The public demo does not claim to reproduce internal thesis experiments.
Experiment-specific code, configurations, and results may be released with a
future thesis, preprint, or archival software version.
