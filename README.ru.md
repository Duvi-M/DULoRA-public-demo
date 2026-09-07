# DULoRA Public Demo

[English](README.md) | [Русский](README.ru.md)

**DULoRA: Data-Dependent Utility-Based Rank Allocation for Parameter-Efficient Transformer Fine-Tuning**

Это публичная, безопасная demo-версия исследовательского проекта DULoRA. Она
показывает инженерную архитектуру, тестируемый Python package и общий workflow
для LoRA-style rank allocation, но не раскрывает неопубликованную
исследовательскую логику.

![Public DULoRA pipeline](assets/public_pipeline.png)

## Что показывает demo

| Область | Содержание |
| --- | --- |
| Проблема | Fixed-rank LoRA назначает одинаковый rank всем выбранным модулям. |
| Идея | Отделить rank-allocation decision от training/evaluation infrastructure. |
| Код | Устанавливаемый `src/` package, PyTorch pipeline, tests, configs и scripts. |
| Allocator | Детерминированный round-robin placeholder только для публичного запуска. |
| Интеграции | Опциональные точки интеграции с Hugging Face Transformers и PEFT. |
| Results | Только безопасные high-level summaries; demo metrics не являются benchmark results. |

## Граница публичной версии

Этот репозиторий намеренно не содержит:

- DULoRA utility estimator.
- Adaptive allocation policy.
- Internal experiment configs.
- Private layer-level assignments.
- Complete unpublished benchmark artifacts.
- Thesis/manuscript figures that are not ready for public release.

Полная английская версия README является canonical landing page:
[README.md](README.md).

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo/run_demo.py
python examples/minimal_text_classification.py
```

Default demo работает offline: используются synthetic token IDs, generated
labels, placeholder layer names, tiny PyTorch classifier и non-research demo
allocator.

## Безопасный summary результатов

| Topic | Public-safe statement |
| --- | --- |
| Model families | Broader research includes BERT-style text classification and ViT-style image classification workflows. |
| PEFT context | LoRA-style parameter-efficient fine-tuning with controlled adapter budgets. |
| Parameter budget | Reported budget-96 setting reduces trainable adapter parameters by 50% compared with fixed LoRA rank 8. |
| Accuracy | Preliminary and dataset-dependent; no universal outperformance claim. |

![Parameter budget comparison](assets/parameter_comparison.png)

Дополнительная документация:

- [Architecture](docs/architecture.md)
- [Methodology overview](docs/methodology_overview.md)
- [Results boundary](docs/results.md)
- [Reproducibility](docs/reproducibility.md)

## Лицензия

Код и публичные безопасные assets в этом репозитории распространяются по
[MIT License](LICENSE). Материалы, не включенные в публичный репозиторий,
остаются вне области этой лицензии.
