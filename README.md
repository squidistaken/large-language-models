# Cross-Lingual Multi-Label Emotion Classification Using Large Language Models

Repository for the Large Language Models course (WBAI068-05) at the University of Groningen.

## Team

 * Petros Kontopirghos (S5851335)
 * Marcus Harald Olof Persson (S5343798)
 * Julian Wierenga (S5628237)

## Development

We use [uv](https://docs.astral.sh/uv/) for project management. Installation instructions are found there.

1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```
3. Clone the [environment](.env.example) as `.env` and fill in the necessary blanks.
4. Run the program.

```bash
uv run main.py
```

## CLI

The project exposes a small Hydra-powered CLI, run via [main.py](main.py). Configuration is split into one group per action under [conf/](conf/), composed by [conf/main.yaml](conf/main.yaml). Override any value on the command line with standard Hydra syntax.

Running without an `action` override launches an interactive menu where you can pick an action from a list. To run an action directly, pass it as an override:

```bash
uv run main.py action=download_data
```

### Downloading Data

Downloads the [XED](https://github.com/Helsinki-NLP/XED) annotated and projection data files for one or more languages.

Interactively, choose "Download Data" from the menu and you'll be prompted for:

- **Languages:** A comma-separated list of language codes, or `all` for every supported
  language.
- **Output directory:** Where the files are saved.
- **Force re-download:** Whether to overwrite files that already exist locally.

Non-interactively, run it directly with Hydra overrides:

```bash
uv run main.py action=download_data data.languages=[en,hr] data.out_dir=data/raw/xed data.force=false
```

Files that already exist in the output directory are skipped unless `data.force=true`.

### Data Preparation

Builds a fixed, leakage-controlled 70/20/10 train/development/test split of **matched
English–target-language pairs** from the raw XED files, plus conversational variants for
fine-tuning.

Interactively, choose "Prepare Data" from the menu and you'll be prompted for:

- **Target language:** The assigned XED language. English is always the
  source side of the pair.
- **Raw data directory:** Where the downloaded XED files live.
- **Output directory:** The base directory for splits. Each language pair gets its own subdirectory.
- **Split ratios:** Three comma-separated fractions summing to 1.0.
- **Random seed:** Seeds the stratified split.
- **Overwrite existing splits:** Whether to replace split files that already exist.

Non-interactively:

```bash
uv run main.py action=prepare_data prepare.target_lang=hr prepare.seed=42 prepare.overwrite=true
```

### Fine-tuning

Fine-tunes the base model with LoRA on the conversational splits from Data Preparation, in one
of three conditions: English only, target language only, or both (bilingual).

Interactively, choose "Fine-tune (PEFT)" from the menu and you'll be prompted for:

- **Condition:** `en`, the target language, or `bilingual`.
- **Model:** One of the configs in [conf/model](conf/model), e.g. `qwen3_5_0_8b` for a quick
  local test.
- **Run name:** The name of the output folder.

Non-interactively:

```bash
uv run main.py action=finetune model=qwen3_5_9b finetune.condition=bilingual finetune.run_name=base
```

On a small GPU, keep the effective batch the same with smaller steps, e.g.
`finetune.batch_size=2 finetune.grad_accum_steps=8`.

### Scoring and Configuration Selection

Scoring lets a fine-tuned run answer every line of a split and computes all the required metrics for the project. Fine-tuning scores the new run on dev by itself; use
this action to score a run again, or on the test set.

Interactively, choose "Score Fine-tuned Run" from the menu and you'll be prompted for:

- **Run folder:** A finished fine-tuning run; the latest one by default.
- **Split:** `dev` or `test`.

Non-interactively:

```bash
uv run main.py action=score_finetune finetune.score.run_dir=outputs/finetune/bilingual/base finetune.score.split=dev
```
