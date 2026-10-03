# Cross-Lingual Multi-Label Emotion Classification Using Large Language Models

Repository for the Large Language Models course (WBAI068-05) at the University of Groningen.

We classify the emotions in movie subtitles from the [XED](https://github.com/Helsinki-NLP/XED)
dataset with open-weight LLMs, in English and our assigned language, Croatian (`hr`). Each line can carry several of Plutchik's eight emotions. We compare zero-shot prompting with parameter-efficient fine-tuning (LoRA/QLoRA) in three conditions: English only, Croatian only, and bilingual.

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
| [`finetune`](conf/finetune/default.yaml) | LoRA and training hyperparameters, scoring, selection. |
| [`analysis`](conf/analysis/default.yaml) | Comparing two systems' predictions. |
| [`wandb`](conf/wandb/default.yaml) | Weights & Biases project and mode. |
| [`logging`](conf/logging/default.yaml) | Log folder and level. |

Models (`model=<name>`):

| Name | Model |
|---|---|
| `eurollm_9b` (default) | [utter-project/EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct) |
| `qwen3_5_9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| `qwen3_5_0_8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |

The global `seed` (42) seeds the data split, training and sampled subsets. Decoding is greedy.

## Actions

### Downloading Data

Downloads the XED annotated and projected data files for one or more languages. Files that
already exist are skipped unless `data.force=true`.

Interactively you are asked for the languages (comma-separated, or `all`), the output folder, and
whether to re-download.

```bash
uv run main.py action=download_data data.languages=[en,hr]
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=download_data data.languages=[en,hr]
```

### Data Preparation

Builds a fixed, leakage-controlled 70/20/10 train/dev/test split of **matched English–Croatian pairs**, plus conversational versions of each split for fine-tuning:

- the split is made by XED alignment id with multi-label stratification, so parallel versions of a subtitle always land in the same partition;
- exact duplicates are removed and never cross partitions;
- Croatian labels are projected from English, and are treated as projected reference labels.

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

Fine-tunes the model with LoRA (QLoRA with `model.quantization=4bit`) in one condition: `en`, `hr`, or `bilingual`. Only the answer tokens are trained. Four times per epoch (`finetune.evals_per_epoch`) the model is evaluated on dev twice: the dev loss, and the metrics of its generated answers on a fixed, seeded subset of 200 dev lines per language (`finetune.metric_samples`; `null` for all, `0` to skip). Both go to W&B and `run_info.json`, so the metrics can be followed during training. A checkpoint is saved at every evaluation; the one with the lowest dev loss is kept (`finetune.checkpoint_metric=micro_f1` keeps the highest dev micro-F1 instead), and training stops after two evaluations without improvement (`finetune.early_stopping_patience`; `null` to always train every epoch). The run then scores itself on the full dev set and reruns the configuration selection (see below).

Interactively you are asked for the condition, the model, and a run name. Use the same run name for all three conditions of one configuration.

```bash
uv run main.py action=finetune finetune.condition=bilingual finetune.run_name=base
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=finetune finetune.condition=bilingual finetune.run_name=base
```

The defaults are rank 16, alpha 32, dropout 0.05 on all linear layers; learning rate 2e-4, 3 epochs, batch 8 with 2 accumulation steps, maximum length 256 tokens, 3% warm-up, seed 42. On a small GPU, keep the effective batch the same with smaller steps. For a quick test, limit the data with `finetune.max_train_samples=64 finetune.max_eval_samples=32 finetune.score.max_samples=32`.

Output: `outputs/hr/finetune/<condition>/<run_name>/`, with the adapter and a `run_info.json` recording every hyperparameter, the model revision, library versions, hardware, peak GPU memory and the training log.

### Scoring and Configuration Selection

Lets a fine-tuned run answer every line of a split and computes all the metrics. Fine-tuning
already scores each run on dev; use this to score a run again, or on test.

After every full dev scoring, the final configuration is selected: the run name with the highest dev micro-F1, averaged over the three conditions per language and then over English and Croatian. All candidates must share the model, data and seed. The ranking is saved to `outputs/hr/finetune/selection.json`.

Interactively you are asked for the run folder (the latest by default) and the split.

```bash
uv run main.py action=score_finetune finetune.score.run_dir=outputs/hr/finetune/bilingual/base finetune.score.split=dev
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=score_finetune finetune.score.run_dir=outputs/hr/finetune/bilingual/base finetune.score.split=dev
```

Scoring a run on test that is not the selected configuration prints a warning: only the selected
configuration's test scores should be reported.

### Comparing Predictions

Compares two systems line by line, for the qualitative analysis. Each line is categorised as improved, worsened, same-correct or same-incorrect by its per-line Jaccard index, and enriched with the aligned English subtitle and its labels, the label source, and the number of alignment hops, to help tell model errors from alignment or projection noise.

Interactively you are asked for the two prediction files.

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
│   ├── <condition>/<run_name>/
│   │   ├── adapter/                  the kept LoRA adapter
│   │   ├── run_info.json             settings, versions, hardware, training log
│   │   └── scores/                   <split>.json and <split>.<lang>.predictions.jsonl
│   └── selection.json                the dev-set ranking and the final configuration
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
