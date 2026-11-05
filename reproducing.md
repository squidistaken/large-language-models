# Reproducing the Project

This guide walks through every run behind the results in our report, in the order we ran them, from an empty checkout to the final results and figures. The [README](README.md) explains each action on its own. This file says which runs to make, with which settings, and where their outputs end up.

Every step gives the command to run locally, followed by its Slurm variant for the Hábrók cluster. The Slurm variant is the same command with `uv run main.py` replaced by `./slurm/run.sh --slurm-config <profile>`.

The steps are:

1. Setup
2. Weights & Biases logging (optional)
3. Preparing the data
4. Prompting on the development set
5. The configuration search
6. Repeating the selected configuration
7. Prompting Qwen3.5-9B on the test set
8. The final evaluation

## What is reproduced

- **Fine-tuning:** QLoRA adapters on EuroLLM-9B-Instruct in three conditions (English only, Croatian only and bilingual), chosen from six candidate configurations on the development set.
- **Prompting baselines:** EuroLLM-9B (4-bit) and Qwen3.5-9B, zero-shot and five-shot.
- **Prompt sensitivity:** two reworded instructions and a random set of demonstrations, scored on the development set only.
- **Robustness:** the selected configuration repeated with two more seeds, and a learning curve over the number of Croatian training lines.
- **Analysis:** per-emotion scores, figures and a line-by-line comparison of zero-shot prompting against the bilingual adapter.

The data split, training and demonstration selection are seeded (seed 42), decoding is greedy, training runs in deterministic mode, both models are pinned to a fixed Hugging Face revision, and `uv.lock` pins every library version.

Everything runs on one NVIDIA A100 (40 GB). A fine-tuning run takes about 0.7 to 0.9 GPU hours, so the 18 runs of the configuration search come to about 13.6 GPU hours, and the 14 repetition runs to at most another 11, as the learning-curve runs are shorter. One prompting pass over the development set takes about four to six minutes. In total, plan for about 30 GPU hours.

## 1. Setup

You need Python 3.13, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for fine-tuning and scoring. Install the dependencies and create your `.env`:

```bash
uv sync
cp .env.example .env
```

In `.env`, set `HF_TOKEN` to a Hugging Face access token so the models can be downloaded. You can leave `WANDB_MODE=disabled` as it is. Weights & Biases is optional (step 2).

**On Hábrók**, clone the repository on a login node and build the environment once (and again after any dependency change):

```bash
./slurm/setup_uv.sh
```

The 9B models and outputs take up a lot of space, so it is best to keep them on `/scratch`. Jobs cache models in `/scratch/$USER/hf-cache` by default. To move the outputs there too, add `out_dir=/scratch/<s-number>/outputs/hr` to every command below.

Two options of `slurm/run.sh` are useful throughout:

- `--dry-run` prints the `sbatch` command without submitting it;
- `--dependency afterok:<id>:<id>` holds a job until the listed jobs have succeeded, so a whole stage can be queued at once.

Follow your jobs with `squeue --me`. Their logs are written to `slurm/logs/`.

## 2. Weights & Biases logging (optional)

W&B is useful for following training as it runs: each adapter gets one run with its training curves, and later scoring jobs add their metrics, per-emotion tables and predictions to that same run.

To enable it, set these in `.env`:

```bash
WANDB_MODE=online          # or offline, to log locally and sync later with `wandb sync`
WANDB_API_KEY=<your key>   # or run `wandb login` once instead
WANDB_ENTITY=<team>        # optional: log to the team's project rather than your own
WANDB_PROJECT=large-language-models
```

## 3. Preparing the data

Download the English and Croatian XED files, then build the fixed 70/20/10 split of matched English–Croatian pairs, along with the conversational versions used for fine-tuning.

```bash
uv run main.py action=download_data data.languages=[en,hr]
uv run main.py action=prepare_data prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=download_data data.languages=[en,hr]
./slurm/run.sh --slurm-config cpu --dependency afterok:<download job id> action=prepare_data prepare.target_lang=hr
```

Output: `data/processed/xed/en-hr/`, with the splits, their conversational versions and a `split_manifest.json` recording every setting.

## 4. Prompting on the development set

Both models are prompted on the development set with seven variants each: zero-shot and five-shot with each of the three instruction templates (`default`, `terse`, `annotator`), plus five-shot with randomly drawn demonstrations instead of the coverage-based set. EuroLLM-9B is prompted in 4-bit, the same precision as its QLoRA adapters. Qwen3.5-9B runs at its default bfloat16.

```bash
for model in "model=eurollm_9b model.quantization=4bit" "model=qwen3_5_9b"; do
  for shots in 0 5; do
    for template in default terse annotator; do
      uv run main.py action=prompting $model prompting.split=dev prompting.shots=$shots \
          prompting.template=$template prepare.target_lang=hr
    done
  done
  uv run main.py action=prompting $model prompting.split=dev prompting.shots=5 \
      prompting.demonstration_selection=random prepare.target_lang=hr
done
```

Slurm:

```bash
for model in "model=eurollm_9b model.quantization=4bit" "model=qwen3_5_9b"; do
  for shots in 0 5; do
    for template in default terse annotator; do
      ./slurm/run.sh --slurm-config a100 action=prompting $model prompting.split=dev \
          prompting.shots=$shots prompting.template=$template prepare.target_lang=hr
    done
  done
  ./slurm/run.sh --slurm-config a100 action=prompting $model prompting.split=dev prompting.shots=5 \
      prompting.demonstration_selection=random prepare.target_lang=hr
done
```

Output: `outputs/hr/prompting/<condition>/<model>/` for each run, with `run_info.json` (the prompt, decoding settings and demonstrations) and `scores/`, holding `dev.json` and the per-line predictions `dev.<lang>.predictions.jsonl`. The condition is `zero_shot` or `5_shot`, plus `-terse`, `-annotator` or `-random` for a variant; the model folder is `eurollm_9b.4bit` or `qwen3_5_9b`.

Once all 14 runs have finished, collect them into one table with bootstrap intervals and the prompt sensitivity:

```bash
uv run main.py action=summarise_prompting prompting.split=dev \
    "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=summarise_prompting prompting.split=dev \
    "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Output: `outputs/hr/prompting/summary.dev.json`, and `prompting.dev` and `prompt_sensitivity.dev` in `outputs/hr/figures/`.

## 5. The configuration search

We compared six fine-tuning configurations, each a preset in [conf/finetune](conf/finetune): LoRA and QLoRA, each at learning rates of 1e-4, 2e-4 (the default) and 4e-4. Each configuration is trained in all three conditions, which makes 18 runs. The preset name doubles as the run name, which groups the three conditions of a configuration together.

```bash
for preset in lora lora_lr1e-4 lora_lr4e-4 qlora qlora_lr1e-4 qlora_lr4e-4; do
  for condition in en hr bilingual; do
    uv run main.py action=finetune finetune=$preset finetune.condition=$condition prepare.target_lang=hr
  done
done
```

Slurm (the 18 jobs are independent and can run in parallel):

```bash
for preset in lora lora_lr1e-4 lora_lr4e-4 qlora qlora_lr1e-4 qlora_lr4e-4; do
  for condition in en hr bilingual; do
    ./slurm/run.sh --slurm-config a100 action=finetune finetune=$preset \
        finetune.condition=$condition prepare.target_lang=hr
  done
done
```

Output: `outputs/hr/finetune/<condition>/<preset>/`, with the kept adapter in `adapter/`, a `run_info.json` recording every setting, the hardware and the training log, and `scores/`, holding `dev.json` and `dev.<lang>.predictions.jsonl`. The dev-set ranking is in `outputs/hr/finetune/selection.json`.

## 6. Repeating the selected configuration

The selected configuration is QLoRA at a learning rate of 1e-4 (`qlora_lr1e-4`), which the `selected_*` presets in [conf/finetune](conf/finetune) repeat. Two sets of extra runs test how robust it is.

 1. **Seeds:** All three conditions are trained again with seeds 67 and 128 (the main runs used 42).
 2. **Learning curve:** The Croatian-only and bilingual conditions are trained on 250, 500, 1,000 and 2,000 Croatian lines. The bilingual runs keep all the English lines. The English-only condition
uses no Croatian training data, so it has no learning curve.

```bash
for preset in selected_seed67 selected_seed128; do
  for condition in en hr bilingual; do
    uv run main.py action=finetune finetune=$preset finetune.condition=$condition prepare.target_lang=hr
  done
done

for preset in selected_n250 selected_n500 selected_n1000 selected_n2000; do
  for condition in hr bilingual; do
    uv run main.py action=finetune finetune=$preset finetune.condition=$condition prepare.target_lang=hr
  done
done
```

Slurm:

```bash
for preset in selected_seed67 selected_seed128; do
  for condition in en hr bilingual; do
    ./slurm/run.sh --slurm-config a100 action=finetune finetune=$preset \
        finetune.condition=$condition prepare.target_lang=hr
  done
done

for preset in selected_n250 selected_n500 selected_n1000 selected_n2000; do
  for condition in hr bilingual; do
    ./slurm/run.sh --slurm-config a100 action=finetune finetune=$preset \
        finetune.condition=$condition prepare.target_lang=hr
  done
done
```

Output: `outputs/hr/finetune/<condition>/qlora_lr1e-4.seed<seed>/` and `outputs/hr/finetune/<condition>/qlora_lr1e-4.n<lines>/`, laid out like the main runs.

## 7. Prompting Qwen3.5-9B on the test set

The final evaluation scores the prompting baselines of the fine-tuned base model (EuroLLM-9B)
itself, but not those of the second model. Score Qwen3.5-9B on the test set separately, zero-shot
and five-shot, with the default prompt only:

```bash
for shots in 0 5; do
  uv run main.py action=prompting model=qwen3_5_9b prompting.split=test prompting.shots=$shots prepare.target_lang=hr
done
```

Slurm:

```bash
for shots in 0 5; do
  ./slurm/run.sh --slurm-config a100 action=prompting model=qwen3_5_9b prompting.split=test \
      prompting.shots=$shots prepare.target_lang=hr
done
```

Output: `outputs/hr/prompting/zero_shot/qwen3_5_9b/` and `outputs/hr/prompting/5_shot/qwen3_5_9b/`, with `test.json` and `test.<lang>.predictions.jsonl` added to `scores/`.

## 8. The final evaluation

This is the only step that uses the test set for the fine-tuned models. It scores the three selected adapters, their seed and learning-curve repetitions, and zero-shot and five-shot EuroLLM-9B (4-bit) on the test set. Systems that have already been scored are reused. It then writes the final results table, every report figure, and the line-by-line comparison of zero-shot prompting against the bilingual adapter. Start it only once steps 5 and 6 are complete.

```bash
uv run main.py action=evaluate_final evaluate.split=test prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config a100 action=evaluate_final evaluate.split=test prepare.target_lang=hr
```

Output: `outputs/hr/results/test.qlora_lr1e-4.json`, with every system's test scores, the seed spread and the learning curve; every figure in `outputs/hr/figures/`, as PDF and PNG; and the comparison in `outputs/hr/analysis/zero_shot-eurollm_9b.4bit_vs_bilingual-qlora_lr1e-4.jsonl`. Each newly scored run also gets `test.json` and its test predictions in its own `scores/` folder.

Then summarise the test-set prompting runs of both models:

```bash
uv run main.py action=summarise_prompting prompting.split=test \
    "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Slurm:

```bash
./slurm/run.sh --slurm-config cpu action=summarise_prompting prompting.split=test \
    "prompting.summary_models=[eurollm_9b.4bit,qwen3_5_9b]" prepare.target_lang=hr
```

Output: `outputs/hr/prompting/summary.test.json`, and `prompting.test` in `outputs/hr/figures/`.
