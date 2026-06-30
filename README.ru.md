# DULoRA Public Demo: динамическое распределение ранга LoRA

[English](README.md) | [Русский](README.ru.md)

### Публичная безопасная demo-версия исследовательского ПО для LoRA workflows

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft/)
[![Tests](https://img.shields.io/badge/tests-8%20passing-brightgreen)](#тестирование)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Master's%20Thesis-purple)](#академический-контекст)

DULoRA — это исследовательский проект магистерской диссертации, посвященный
изучению **неравномерной организации рангов LoRA для parameter-efficient
fine-tuning**.

Этот репозиторий является **публичной demo-версией**. Он сохраняет
профессиональную структуру исследовательского ПО, но публикует только
безопасную и упрощенную реализацию. Цель — показать инженерный workflow вокруг
DULoRA, не раскрывая неопубликованную исследовательскую логику.

> **Этот репозиторий демонстрирует только архитектуру ПО и публичный workflow.
> Он не содержит research-specific scoring DULoRA или адаптивный метод
> распределения.**

Demo построена на PyTorch и содержит опциональные точки интеграции с Hugging
Face Transformers и PEFT. Она использует синтетические/offline данные,
компактный toy-классификатор, публичный интерфейс allocator и детерминированный
demo allocator.

---

## Содержание

- [Мотивация исследования](#мотивация-исследования)
- [Цель исследования](#цель-исследования)
- [Обзор метода](#обзор-метода)
- [Алгоритм](#алгоритм)
- [Адаптивный алгоритм распределения ранга](#адаптивный-алгоритм-распределения-ранга)
- [Обзор pipeline](#обзор-pipeline)
- [Текущие результаты публичной demo](#текущие-результаты-публичной-demo)
- [Сгенерированные demo outputs](#сгенерированные-demo-outputs)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Запуск dashboard](#запуск-dashboard)
- [Запуск эксперимента](#запуск-эксперимента)
- [Как воспроизвести публичную demo](#как-воспроизвести-публичную-demo)
- [Конфигурация](#конфигурация)
- [Optional Transformers and PEFT Integration](#optional-transformers-and-peft-integration)
- [Тестирование](#тестирование)
- [Воспроизводимость](#воспроизводимость)
- [Текущая оценка](#текущая-оценка)
- [Примеры исследовательских вопросов](#примеры-исследовательских-вопросов)
- [Текущие ограничения](#текущие-ограничения)
- [Дальнейшая работа](#дальнейшая-работа)
- [Академический контекст](#академический-контекст)
- [Цитирование](#цитирование)
- [Автор](#автор)
- [Лицензия](#лицензия)

---

## Мотивация исследования

Low-Rank Adaptation, или LoRA, — это parameter-efficient техника fine-tuning,
которая уменьшает количество обучаемых параметров, необходимых для адаптации
больших pretrained моделей к downstream-задачам.

Во многих LoRA workflows один и тот же ранг применяется ко всем выбранным
target modules. Это просто и часто эффективно, но возникает естественный
исследовательский вопрос: всем ли выбранным модулям нужна одинаковая
адаптационная емкость?

DULoRA исследует этот вопрос. Приватная исследовательская версия изучает,
можно ли организовать LoRA capacity более избирательно при контролируемом
бюджете. Эта публичная demo оставляет видимой программную архитектуру, но
намеренно скрывает неопубликованную decision logic.

---

## Цель исследования

Более широкий проект DULoRA направлен на разработку и оценку воспроизводимого
workflow для изучения неравномерной организации рангов LoRA при fine-tuning
transformer-моделей.

Этот публичный репозиторий фокусируется на безопасной инженерной части:

- Чистый устанавливаемый Python package с layout `src/`.
- Модульный LoRA-style allocation interface.
- Offline PyTorch training and evaluation pipeline.
- Опциональные integration points для Transformers/PEFT.
- Малые синтетические fixtures вместо приватных экспериментов или datasets.
- Tests для software contracts, metrics и end-to-end execution.
- Документация публичного workflow без раскрытия неопубликованных деталей
  метода.

Research-specific estimator, adaptive decision policy, внутренние значения
конфигураций и финальные экспериментальные артефакты не включены.

---

## Обзор метода

Приватный research workflow и эта public demo имеют похожую высокоуровневую
форму, но не один и тот же внутренний алгоритм.

### 1. Контекст Baseline LoRA

Исследовательский контекст начинается с обычной LoRA постановки: выбранные
модули модели получают low-rank adapter configuration. Публичная demo описывает
этот контекст концептуально, но не публикует финальные приватные
hyperparameters или benchmark settings.

### 2. Публичный allocator contract

Demo предоставляет стабильный Python interface для allocator-like компонента.
Интерфейс позволяет остальной части pipeline запросить module-to-rank assignment
без знания того, как именно этот assignment получен.

Это полезно для архитектуры ПО: реализации allocator можно менять без
переписывания data, model, trainer или evaluator modules.

### 3. Demonstration allocator

Публичная реализация использует детерминированный round-robin style allocator,
созданный только для этого репозитория. Он намеренно простой:

- Он не анализирует gradients.
- Он не анализирует activations.
- Он не использует losses или training dynamics.
- Он не воспроизводит приватную DULoRA policy.
- Он существует только для того, чтобы публичный pipeline был исполняемым.

---

## Алгоритм

Публичный репозиторий **не публикует исследовательский алгоритм**.

Вместо этого раздел задает границу релиза. Приватная версия содержит
неопубликованную DULoRA decision logic; публичная версия содержит только
software-compatible demonstration path.

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

Эта граница намеренная, пока диссертация и связанная manuscript работа находятся
в подготовке.

---

## Адаптивный алгоритм распределения ранга

Адаптивный исследовательский метод намеренно не включен в эту public demo.
Чтобы проект оставался runnable, репозиторий предоставляет простой
demonstration allocator за похожим интерфейсом.

Публичный allocator API — это software contract, а не раскрытие
исследовательского метода.

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

Возвращаемое назначение детерминированное и синтетическое. Оно безопасно для
examples, tests и portfolio review, но не должно интерпретироваться как
исследовательский алгоритм DULoRA.

---

## Обзор pipeline

Публичный demo pipeline:

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

Приватный research pipeline содержит дополнительные неопубликованные компоненты,
которые намеренно не входят в этот релиз.

---

## Текущие результаты публичной demo

Default run — это offline smoke test, а не академический эксперимент. Он
использует generated token IDs, generated labels, placeholder module names и
публичный demo allocator.

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

Accuracy, F1 и loss, которые печатают scripts, являются **toy runtime
diagnostics**. Это не benchmark results и их нельзя цитировать как research
performance DULoRA.

---

## Сгенерированные demo outputs

Публичные scripts печатают metrics в terminal и рассчитаны на запуск без
создания persistent experiment outputs. Если вы расширяете репозиторий, держите
generated artifacts вне version control.

Рекомендуемые local-only директории:

```text
outputs/
checkpoints/
logs/
wandb/
mlruns/
models/
```

Эти пути игнорируются через `.gitignore`.

---

## Структура проекта

```text
DULoRA-public-demo/
├── assets/
│   └── .gitkeep
├── configs/public/
│   └── demo.yaml
├── docs/
│   ├── methodology_overview.md
│   └── reproducibility.md
├── examples/
│   └── minimal_text_classification.py
├── scripts/demo/
│   └── run_demo.py
├── src/dulora_demo/
│   ├── allocator_interface.py
│   ├── data.py
│   ├── demo_allocator.py
│   ├── evaluator.py
│   ├── model.py
│   ├── reproducibility.py
│   └── trainer.py
├── tests/
│   ├── test_allocator_contract.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── README.md
└── README.ru.md
```

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Для тестов:

```bash
python -m pip install -e ".[dev]"
```

Для optional ML integrations:

```bash
python -m pip install -e ".[ml]"
```

Или все публичные demo features:

```bash
python -m pip install -r requirements.txt
```

---

## Запуск dashboard

Приватный development repository может содержать dashboard-style visualization
для внутренних экспериментальных артефактов. Эта public demo не поставляет
dashboard, потому что не публикует такие артефакты.

Будущий публичный dashboard должен визуализировать только synthetic demo
outputs.

---

## Запуск эксперимента

Configuration-driven demo:

```bash
python scripts/demo/run_demo.py
```

Минимальный package example:

```bash
python examples/minimal_text_classification.py
```

Обе команды используют synthetic inputs и не скачивают datasets или pretrained
models.

---

## Как воспроизвести публичную demo

Из корня репозитория:

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/demo/run_demo.py
python examples/minimal_text_classification.py
python -m compileall -q src examples scripts tests
```

Ожидаемый результат: passing test suite и terminal output со стратегией demo
allocator, synthetic assignment и toy evaluation metrics.

---

## Конфигурация

Публичная конфигурация намеренно маленькая и generic:

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

Эти значения не являются private experiment settings и не предназначены для
воспроизведения thesis results.

---

## Optional Transformers and PEFT Integration

Optional builder показывает стандартную интеграцию с заранее заданным public
или synthetic assignment. Он не выводит research assignment.

```python
from dulora_demo.model import build_peft_sequence_classifier

model = build_peft_sequence_classifier(
    model_name="distilbert-base-uncased",
    num_labels=2,
    rank_pattern=None,
    target_modules=("q_lin", "v_lin"),
)
```

Требуется:

```bash
python -m pip install -e ".[ml]"
```

Вызов builder может скачать third-party artifacts с Hugging Face.

---

## Тестирование

```bash
pytest -q
```

Tests покрывают:

- Allocator contract behavior.
- Deterministic demo assignment.
- Seed control.
- Accuracy и binary F1.
- End-to-end offline training.

Tests не кодируют, не аппроксимируют и не проверяют приватный research method.

---

## Воспроизводимость

Demo entry points устанавливают global seed до создания модели и training.
Helper покрывает Python, PyTorch и CUDA seeding, если CUDA доступна. Также
запрашивается best-effort deterministic PyTorch behavior.

Точная воспроизводимость все еще может зависеть от hardware, backend behavior,
operating system, CUDA/cuDNN settings и library versions.

См. [docs/reproducibility.md](docs/reproducibility.md).

---

## Текущая оценка

Public demo сообщает:

- Training loss.
- Accuracy.
- Binary F1.
- Demo allocator strategy.
- Synthetic assignment summary.

Эти значения печатаются только для smoke testing. Они не являются published
results и не служат доказательством за или против private research hypothesis.

---

## Примеры исследовательских вопросов

Полный research project связан с вопросами:

- Можно ли организовать LoRA capacity более избирательно, чем uniform setup?
- Как корректно контролировать trainable parameter budget при fair comparison?
- Насколько стабильны non-uniform assignments across seeds, tasks и model
  families?
- Какие evaluation protocols лучше подходят для parameter-efficient adaptation
  research?
- Как публиковать достаточно структуры для reproducibility, не раскрывая
  преждевременно unpublished intellectual property?

Этот публичный репозиторий поддерживает только software-engineering сторону
этих вопросов.

---

## Текущие ограничения

- Используются synthetic/offline data.
- По умолчанию используется compact toy classifier.
- Thesis experiments и benchmark results не публикуются.
- Private method details не включены.
- Final ablations, internal configs и layer-level research artifacts не
  предоставляются.
- Optional Transformers/PEFT integration — это safe extension point, а не full
  reproduction package.

---

## Дальнейшая работа

Возможные публичные extensions:

- Добавить больше toy tasks, которые остаются offline и safe.
- Опубликовать дополнительную документацию по experiment hygiene.
- Добавить tests для package installation и CLI behavior.
- Добавить dashboard только для synthetic demo outputs.
- Выпустить formal reproduction package после thesis/manuscript process.
- Позже опубликовать research-specific components в подходящем archival
  release.

---

## Академический контекст

Этот репозиторий создан как public-safe companion к магистерскому thesis project
по parameter-efficient fine-tuning transformer-моделей.

Public release предназначен для portfolio review, research-software inspection
и reproducibility practice. Это не полная research implementation DULoRA.

---

## Цитирование

Citation metadata находится в [CITATION.cff](CITATION.cff).

До появления официальной статьи, thesis record или preprint можно цитировать
репозиторий как software demonstration:

```text
Duvan Mendoza. DULoRA Public Demo, version 0.1.0, 2026.
https://github.com/Duvi-M/DULoRA-public-demo
```

---

## Автор

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

## Лицензия

Код и public-safe assets в этом репозитории распространяются по
[MIT License](LICENSE).

Лицензия применяется только к материалам, включенным в этот публичный
репозиторий. Research components, которые здесь не распространяются, находятся
за пределами этой лицензии.
