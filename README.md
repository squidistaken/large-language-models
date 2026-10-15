# Cross-Lingual Multi-Label Emotion Classification Using Large Language Models

Repository for the Large Language Models course (WBAI068-05) at the University of Groningen.

We classify the emotions in movie subtitles from the [XED](https://github.com/Helsinki-NLP/XED)
dataset with open-weight LLMs, in English and our assigned language, Croatian (`hr`). Each line can carry several of Plutchik's eight emotions. We compare zero-shot prompting with parameter-efficient fine-tuning (LoRA/QLoRA) in three conditions: English only, Croatian only, and bilingual.

## Model cards

The three QLoRA adapters (EuroLLM-9B-Instruct) each have a model card with the prompt template, training settings, results and limitations:

- [English only](model_cards/en.md)
- [Croatian only](model_cards/hr.md)
- [Bilingual](model_cards/bilingual.md)

## Team

- Petros Kontopirghos (S5851335)
- Marcus Harald Olof Persson (S5343798)
- Julian Wierenga (S5628237)

## Setup

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for fine-tuning and
scoring. We run the 9B models on an A100 (40 GB); `model.quantization=4bit` (QLoRA) lowers the
memory needed.

```bash
uv sync
cp .env.example .env    # then fill in the values below
uv run main.py          # opens the interactive menu
```

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access token, for downloading the models. |
| `HF_HOME` | Optional model cache location; on Hábrók, jobs default to `/scratch/$USER/hf-cache`. |
| `WANDB_MODE` | `online`, `offline` or `disabled` (an empty value counts as disabled). |
| `WANDB_API_KEY` | Needed for online W&B runs, unless you have run `wandb login`. |
| `WANDB_ENTITY`, `WANDB_PROJECT` | The W&B team and project to log to; the entity defaults to your own. |
| `WANDB_DIR`, `UV_CACHE_DIR` | Optional cache locations, e.g. on `/scratch`. |

Model weights, access tokens and the dataset are never committed: `data/`, `outputs/`, `logs/`,
`wandb/` and `.env` are git-ignored.

## Configuration

The CLI is run through [main.py](main.py) with [Hydra](https://hydra.cc). Running it without an `action` opens an interactive menu; passing `action=<name>` runs that action directly. Any setting can be overridden on the command line.

[conf/main.yaml](conf/main.yaml) composes one config group per concern:

| Group | Settings |
|---|---|
| [`data`](conf/data/default.yaml) | Languages to download, and where to. |
| [`prepare`](conf/prepare/default.yaml) | Target language, split ratios, split seed. |
| [`model`](conf/model) | The model: Hugging Face id, pinned revision, dtype, quantization, decoding. |
| [`finetune`](conf/finetune) | LoRA and training hyperparameters, scoring, selection; plus named PEFT configurations. |
| [`analysis`](conf/analysis/default.yaml) | Comparing two systems' predictions. |
| [`wandb`](conf/wandb/default.yaml) | Weights & Biases project and mode. |
| [`logging`](conf/logging/default.yaml) | Log folder and level. |

Models (`model=<name>`):

| Name | Model |
|---|---|
| `eurollm_9b` (default) | [utter-project/EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) |
| `qwen3_5_9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `qwen3_5_0_8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| `smollm2_135m` | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |

The global `seed` (42) seeds the data split, training and sampled subsets. Decoding is greedy.

## Actions

### Downloading Data

Downloads the XED annotated and projected data files for one or more languages. Files that
already exist are skipped unless `data.force=true`.

Interactively you are asked for the languages (comma-separated, or `all`; `en,hr` by default), the
output folder, and whether to re-download.

```bash
uv run main.py action=download_data data.languages=[en,hr]
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=download_data data.languages=[en,hr]
```

### Data Preparation

Builds a fixed, leakage-controlled 70/20/10 train/dev/test split of **matched English–target-language pairs** (English–Croatian by default), plus conversational versions of each split for fine-tuning:

- the split is made by XED alignment id with multi-label stratification, so parallel versions of a subtitle always land in the same partition;
- exact duplicates are removed and never cross partitions;
- target-language labels are projected from English, and are treated as projected reference labels.

Interactively you are asked for the target language, folders, split ratios, seed, and whether to overwrite. Existing splits are reused; if they were built with other settings the action stops, and `prepare.overwrite=true` rebuilds them.

```bash
uv run main.py action=prepare_data prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=prepare_data prepare.target_lang=hr
```

Output: `data/processed/xed/en-hr/`, with a `split_manifest.json` recording every setting.

### Fine-tuning

Fine-tunes the model with LoRA (QLoRA with `model.quantization=4bit`) in one condition for the target language (`prepare.target_lang`, Croatian by default):

| Condition | Trains on | Evaluated on |
|---|---|---|
| `en` | English | English and the target language |
| `<target>` | the target language | the target language |
| `bilingual` | English + the target language | English and the target language |

Interactively you are asked for the target language, then shown what each condition trains and is evaluated on, and asked for the condition, the model, and a run name. Use the same run name for all three conditions of one configuration.

```bash
uv run main.py action=finetune finetune.condition=bilingual finetune.run_name=base
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=finetune finetune.condition=bilingual finetune.run_name=base
```

The defaults are rank 16, alpha 32, dropout 0.05 on all linear layers; learning rate 2e-4, 3 epochs, batch 8 with 2 accumulation steps, maximum length 256 tokens, 3% warm-up, seed 42. The model is evaluated on dev twice per epoch; the checkpoint with the highest dev micro-F1 is kept, and training stops after one epoch without improvement. On a small GPU, keep the effective batch the same with smaller steps. For a quick test, limit the data with `finetune.max_train_samples=64 finetune.max_eval_samples=32 finetune.score.max_samples=32`.

Named configurations are saved as presets next to the defaults in [conf/finetune](conf/finetune) and chosen with `finetune=<preset>`.

Output: `outputs/hr/finetune/<condition>/<run folder>/`, with the adapter and a `run_info.json` recording every hyperparameter, the model revision, library versions, hardware, peak GPU memory and the training log.

### Scoring and Configuration Selection

Lets a fine-tuned run answer every line of a split and computes all the metrics. Fine-tuning
already scores each run on dev; use this to score a run again, or on test.

After every full dev scoring, one final PEFT configuration is selected. A configuration is the set of
fine-tuning choices the three conditions share (LoRA or QLoRA, adapter, optimisation and checkpoint
settings), trained under one run name; the base model, data split and seed are the same for every
candidate. Each candidate's dev micro-F1 is averaged per language over the conditions evaluated in
that language (English: `en` and `bilingual`; the target language: all three), then over the two
languages with equal weight, and the highest score wins. Only the selected configuration is scored on
test.

Interactively you are asked for the target language, the run folder (the latest of that language by default) and the split.

```bash
uv run main.py action=score_finetune finetune.score.run_dir=outputs/hr/finetune/bilingual/base finetune.score.split=dev
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=score_finetune finetune.score.run_dir=outputs/hr/finetune/bilingual/base finetune.score.split=dev
```

Output: `outputs/hr/finetune/<condition>/<run_name>/scores/`, with `<split>.json` and the per-line predictions `<split>.<lang>.predictions.jsonl`, and the dev-set ranking in `outputs/hr/finetune/selection.json`.

### Prompting

Lets the base model answer every line of a split, zero-shot or with demonstrations, and computes all the metrics. With `prompting.shots=5`, five demonstrations are chosen from the train split: short gold English lines with a direct alignment, one per film, covering as many emotions as possible (seeded). English is prompted with the English lines and the target language with their translations. `prompting.demonstration_ids` pins a set instead.

Two variants serve the prompt-sensitivity analysis and are scored on dev only:

1. `prompting.template` picks the instruction: `default`, `terse` or `annotator`. The wording lives in [conf/prompting/default.yaml](conf/prompting/default.yaml), where you can add templates. `default` is the prompt the PEFT runs were trained with and cannot be changed without preparing the splits again.
2. `prompting.demonstration_selection=random` draws the demonstrations at random instead.

Prompt the primary model with the quantization of its PEFT runs (`model.quantization=4bit` for QLoRA).

Interactively you are asked for the target language, the model, the split, the number of shots, the demonstration selection and the template.

```bash
uv run main.py action=prompting model=eurollm_9b model.quantization=4bit prompting.shots=5 prompting.split=dev prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=prompting model=eurollm_9b model.quantization=4bit prompting.shots=5 prompting.split=dev prepare.target_lang=hr
```

Output: `outputs/hr/prompting/<condition>/<model>/`, with the scores, the per-line predictions and a `run_info.json` recording the prompt, the decoding settings and the demonstrations. The condition is `zero_shot` or `5_shot`, plus `-random` or `-<template>` for a variant; the model folder ends in `.4bit` when quantized.

### Summarising Prompting Runs

Collects every prompting run scored on a split into one table, with bootstrap confidence intervals, and adds the prompt sensitivity: the mean, standard deviation and range of micro-F1 and Jaccard across the templates.

Interactively you are asked for the target language, the split and the models.

```bash
uv run main.py action=summarise_prompting prompting.split=dev "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=summarise_prompting prompting.split=dev "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Output: `outputs/hr/prompting/summary.<split>.json`, and `prompting.<split>` and `prompt_sensitivity.<split>` in `outputs/hr/figures/`.

### Final Evaluation

Scores the selected PEFT configuration in all three conditions on a split (test by default), next to the prompting baselines of its base model, set by `evaluate.prompt_shots` (`[0, 5]` by default). Systems already scored are reused. It also reports the spread over seeds, the learning curve, all figures and a line-by-line comparison of zero-shot against the bilingual run.

Interactively you are asked for the target language, the split and the prompting baselines.

```bash
uv run main.py action=evaluate_final evaluate.split=test prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=evaluate_final evaluate.split=test prepare.target_lang=hr
```

Output: `outputs/hr/results/<split>.<run_name>.json`, the figures in `outputs/hr/figures/` and the comparison in `outputs/hr/analysis/`.

### Comparing Predictions

Compares two systems line by line, for the qualitative analysis. Each line is categorised as improved, worsened, same-correct or same-incorrect by its per-line Jaccard index, and enriched with the aligned English subtitle and its labels, the label source, and the number of alignment hops, to help tell model errors from alignment or projection noise.

Interactively you are asked for the target language and the two prediction files.

```bash
uv run main.py action=compare_predictions analysis.baseline=outputs/hr/finetune/en/base/scores/test.hr.predictions.jsonl analysis.candidate=outputs/hr/finetune/hr/base/scores/test.hr.predictions.jsonl
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=compare_predictions analysis.baseline=outputs/hr/finetune/en/base/scores/test.hr.predictions.jsonl analysis.candidate=outputs/hr/finetune/hr/base/scores/test.hr.predictions.jsonl
```

Output: `outputs/hr/analysis/<name>.jsonl`, one row per line.

## Outputs

```text
data/
├── raw/xed/                          downloaded XED files
└── processed/xed/en-hr/              splits, conversational splits, split_manifest.json
outputs/<lang>/                       one folder per target language (out_dir)
├── finetune/
│   ├── <condition>/<run_name>[.seed<seed>|.n<lines>]/
│   │   ├── adapter/                  the kept LoRA adapter
│   │   ├── run_info.json             settings, versions, hardware, training log
│   │   └── scores/                   <split>.json and <split>.<lang>.predictions.jsonl
│   └── selection.json                the dev-set ranking and the final configuration
├── prompting/
│   ├── <condition>/<model>/          run_info.json (prompt, demonstrations) and scores/
│   └── summary.<split>.json          all prompting runs and the prompt sensitivity
├── results/<split>.<run_name>.json   the final results table, seeds and learning curve
├── figures/                          the report figures, PDF and PNG
└── analysis/<name>.jsonl             prediction comparisons
logs/                                 one log file per CLI run
```

Each target language keeps its own outputs, so runs for different languages never overwrite each
other: `prepare.target_lang=sv` writes to `outputs/sv/`. Move every output elsewhere, e.g. to
scratch storage, with `out_dir=/scratch/<s-number>/outputs/hr`.

## Metrics

- **Primary:** micro-averaged F1 and sample-averaged Jaccard index.
- **Secondary:** macro-averaged F1, micro-averaged precision and recall, Hamming loss (lower is
  better).
- **Diagnostic:** per-emotion precision, recall, F1 and support; exact-match ratio; malformed and
  empty output rates.

Malformed answers, and answers with no permitted label, count as an empty label set, and the malformed rate is reported separately. A metric whose denominator is zero is set to 0.0. 

## Weights & Biases

Set `WANDB_MODE=online` and `WANDB_API_KEY` in `.env` (or run `wandb login`). The project defaults to `large-language-models` under your own entity; set `WANDB_ENTITY` and `WANDB_PROJECT` in `.env` to log to the team's project instead.

Each fine-tuned adapter is one W&B run (`<condition>/<run_name>`, grouped by run name). It holds the training curves, and later scoring jobs resume the same run, adding the metrics, per-emotion tables and predictions.

## Hábrók (Slurm)

Every action can run as a job: replace `uv run main.py` with
`./slurm/run.sh --slurm-config <profile>` and keep the overrides. Add `--dry-run` to print the
`sbatch` command without submitting, or `--dependency afterok:<id>:<id>` to start a job only once
others have succeeded.

| Profile | Resources | For |
|---|---|---|
| `cpu` | 4 CPUs, 1 h, no GPU | downloading and preparing data, comparing predictions |
| `smoke` | A100, 30 min | quick tests with the small model |
| `a100` | A100 (40 GB), 8 h | fine-tuning and scoring |
| `rtx` | RTX Pro 6000 (96 GB), 8 h | the same, with more memory; needs `.venv-rtx` |

Profiles live in [slurm/conf/](slurm/conf). Set up the environment once on a login node, and
again after dependency changes:

```bash
./slurm/setup_uv.sh                # .venv; add --use-rtx to build .venv-rtx in a GPU job
```

Follow jobs with `squeue --me`; their logs are in `slurm/logs/`.

## Development

```bash
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run ty check                                       # type check
uv run pre-commit install                             # run both on every commit
```
