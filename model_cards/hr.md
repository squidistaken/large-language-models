---
base_model: utter-project/EuroLLM-9B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language: [hr]
datasets: [Helsinki-NLP/XED]
tags: [lora, qlora, peft, multi-label-classification, emotion]
---

# EuroLLM-9B XED emotion QLoRA: Croatian only

A QLoRA adapter for EuroLLM-9B-Instruct that labels the emotions in Croatian movie subtitle lines. It was trained on the Croatian XED training split only.

## Model details

- Base model: [utter-project/EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct), revision `f7ae2bc3bcbb538c0b93fa6cfbf388d9898b1ace` (Apache 2.0).
- Adapter: PEFT LoRA on a 4-bit NF4 base model (QLoRA), for causal language modelling.
- Authors: Petros Kontopirghos, Marcus Persson and Julian Wierenga, for the Large Language Models course (WBAI068-05) at the University of Groningen.
- Code: https://github.com/squidistaken/large-language-models

## Intended use

This is for research on cross-lingual, multi-label emotion classification of movie subtitle lines, using Plutchik's eight emotions. It has not been tested on anything else, and it should not be used to make decisions about people.

## Prompt template

The adapter was trained and tested with this system message and user message. The model answers with a JSON list of label names.

```text
[system]
You classify the emotions expressed in a movie subtitle line.

Permitted labels:
- anger: displeasure or hostility at a perceived wrong
- anticipation: expecting or looking forward to something
- disgust: revulsion or strong disapproval
- fear: feeling threatened or afraid
- joy: happiness or pleasure
- sadness: sorrow, loss or disappointment
- surprise: reaction to something unexpected
- trust: confidence in, or acceptance of, someone or something

Every line expresses at least one of these emotions and may express several. Answer with a JSON list of the matching label names, using only the names above, and nothing else, in the format ["label", ...].

[user]
Subtitle: {text}

[assistant]
["label", ...]
```

Decoding is greedy with `max_new_tokens=32`. An answer that cannot be parsed, or has no permitted label, counts as an empty prediction.

## Reproduction

The weights are not published, so the adapter has to be trained again with the code in the repository (see the [README](https://github.com/squidistaken/large-language-models#setup) for the setup). You need a CUDA GPU; we used one A100 (40 GB), which the `a100` Slurm profile requests.

On a machine with a GPU:

```bash
uv sync
uv run main.py action=download_data data.languages=[en,hr]
uv run main.py action=prepare_data prepare.target_lang=hr
uv run main.py action=finetune finetune=qlora_lr1e-4 finetune.condition=hr prepare.target_lang=hr
uv run main.py action=score_finetune finetune.score.run_dir=outputs/hr/finetune/hr/qlora_lr1e-4 finetune.score.split=test
```

On Hábrók, with Slurm jobs (profiles are in `slurm/conf/`; `--dry-run` prints the `sbatch` command without submitting):

```bash
./slurm/setup_uv.sh    # once, on a login node
./slurm/run.sh --slurm-config cpu action=download_data data.languages=[en,hr]
./slurm/run.sh --slurm-config cpu --dependency afterok:<download job id> action=prepare_data prepare.target_lang=hr
./slurm/run.sh --slurm-config a100 --dependency afterok:<prepare job id> action=finetune finetune=qlora_lr1e-4 finetune.condition=hr prepare.target_lang=hr
./slurm/run.sh --slurm-config a100 --dependency afterok:<finetune job id> action=score_finetune finetune.score.run_dir=outputs/hr/finetune/hr/qlora_lr1e-4 finetune.score.split=test
```

The settings below are the `qlora_lr1e-4` preset in `conf/finetune/`. The adapter ends up in `outputs/hr/finetune/hr/qlora_lr1e-4/`, next to a `run_info.json` with every setting.

## Training data

The Croatian XED training split: 3,802 lines with the projected labels provided by [XED](https://github.com/Helsinki-NLP/XED) (1,068 dev and 551 test lines). The prompt stays in English, only the subtitle line is Croatian.

## Training

| Setting | Value |
|---|---|
| Adapter targets | all linear layers |
| Rank $r$ / $\alpha$ / dropout | $16$ / $32$ / $0.05$ |
| Quantization | 4-bit NF4 with double quantization, bfloat16 compute |
| Learning rate | $10^{-4}$, warm-up ratio 0.03 |
| Batch size | $8 \times 2$ accumulation steps $= 16$ |
| Epochs (maximum / trained) | 3 / 3 |
| Maximum sequence length | 256 tokens |
| Checkpoint rule | dev micro-F1 twice per epoch; keep the best; stop after 2 evaluations without improvement |
| Seed | 42 (67 and 128 for the seed comparison) |
| Hardware | $1 \times$ NVIDIA A100-PCIE-40GB |
| Trainable parameters | 50,921,472 of 9,203,240,960 ($0.55\%$) |
| Peak GPU memory / training time | 13.81 GiB / 0.83 h |
| Software | Python 3.13.12, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.2, scikit-learn 1.9.1 |

We picked `qlora_lr1e-4` out of six candidates (LoRA and QLoRA, with learning rates $10^{-4}$, $2 \times 10^{-4}$ and $4 \times 10^{-4}$). The choice used the dev set only: the dev micro-F1 averaged over the three conditions, then over English and Croatian. It scored 0.479, against 0.477 for the runner-up.

## Evaluation

Results on the test set, 551 lines per language, with 95% bootstrap confidence intervals in brackets.

| Language | Micro-F1 | Jaccard (samples) | Macro-F1 | Exact match |
|---|---|---|---|---|
| hr | 0.403 [0.369, 0.434] | 0.359 [0.325, 0.393] | 0.380 [0.346, 0.410] | 0.227 |

Without any training, the same base model (EuroLLM-9B, 4-bit) gets a micro-F1 of 0.380 on Croatian with zero-shot prompting and 0.395 with five shots. The adapter reaches 0.403, but the five-shot interval overlaps with it, so we have not shown a gain over five-shot prompting. Qwen3.5-9B with no training scores 0.422 zero-shot and 0.444 five-shot on Croatian, which is higher than this adapter.

Over three seeds (42, 67, 128) the micro-F1 on Croatian is $0.4025 \pm 0.0002$. We did not evaluate this adapter on English.

## Limitations

On Croatian it scores below the bilingual adapter (0.437), and we did not evaluate it on English.

The Croatian labels are the projected labels provided by [XED](https://github.com/Helsinki-NLP/XED), not native Croatian annotations, so projection and alignment errors can limit the score, and the test set has the same noise.

The test set is small (551 lines per language). The confidence intervals are about $\pm 3$ points wide, so smaller differences should not be over-interpreted, and the micro-F1 standard deviation over seeds was up to 0.02.

The data is movie subtitles only: short lines with no context. We did not test other kinds of text. The emotion labels are also subjective, so a disagreement with the gold label is not always a model error.

The adapter also carries over the limitations of EuroLLM-9B-Instruct. We make no claims that differences between the languages are cultural.
