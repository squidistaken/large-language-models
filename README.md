# Cross-Lingual Multi-Label Emotion Classification Using Large Language Models

Repository for the Large Language Models course (WBAI068-05) at the University of Groningen.

## Team

 * Petros Kontopirghos (S5851335)
 * Marcus Harald Olof Persson (S5343798)
 * Julian Wierenga (S)

## Development

We use [uv](https://docs.astral.sh/uv/) for project management. Installation instructions are found there.

1. Clone the project.
2. Synchronise the project.

```bash
uv sync
```
3. Run the program.

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
of three conditions: English only, target language only, or both (**bilingual**).

Interactively, choose "Fine-tune (PEFT)" from the menu and you'll be prompted for:

- **Condition:** `en`, the target language, or `bilingual`.
- **Hugging Face model:** The base model to fine-tune.
- **Run name:** The name of the output folder.

Non-interactively:

```bash
uv run main.py action=finetune finetune.model_name=<hf-model> finetune.condition=bilingual finetune.run_name=base
```

All hyperparameters are set in [conf/finetune/default.yaml](conf/finetune/default.yaml). The
adapter and a `run_info.json` with the full configuration are saved to
`outputs/finetune/<condition>/<run_name>/`. Existing runs are not replaced unless
`finetune.overwrite=true`.

### Scoring

Lets a fine-tuned run answer every line of a split and computes micro-F1, Jaccard and the
malformed-output rate for each language.

Interactively, choose "Score Fine-tuned Run" from the menu and you'll be prompted for:

- **Run folder:** A finished fine-tuning run.
- **Split:** `dev` or `test`.

Non-interactively:

```bash
uv run main.py action=score_finetune finetune.score.run_dir=outputs/finetune/bilingual/base finetune.score.split=dev
```

Scores and per-line predictions are saved to the run's `scores/` folder.

### Configuration Selection

Picks one configuration for all three conditions by mean development micro-F1, with English and
the target language weighted equally.

Interactively, choose "Select Final PEFT Configuration" from the menu and you'll be prompted for:

- **Candidate run names:** A comma-separated list of runs that have been fine-tuned and scored
  in every condition.

Non-interactively:

```bash
uv run main.py action=select_finetune 'finetune.select.candidates=[base,lr_low,rank8]'
```

The ranking and the selected configuration are saved to `outputs/finetune/selection.json`.

### Reproducing the PEFT Results

Train and score each candidate in every condition, then select on the development set:

```bash
MODEL=<hf-model>

for cond in en hr bilingual; do
  uv run main.py action=finetune finetune.model_name=$MODEL finetune.condition=$cond finetune.run_name=base
  uv run main.py action=finetune finetune.model_name=$MODEL finetune.condition=$cond finetune.run_name=lr_low finetune.learning_rate=1e-4
  uv run main.py action=finetune finetune.model_name=$MODEL finetune.condition=$cond finetune.run_name=rank8 finetune.lora.r=8 finetune.lora.alpha=16

  for run in base lr_low rank8; do
    uv run main.py action=score_finetune finetune.score.run_dir=outputs/finetune/$cond/$run
  done
done

uv run main.py action=select_finetune 'finetune.select.candidates=[base,lr_low,rank8]'
```

Finally, score the selected runs on the test set with `finetune.score.split=test`.




