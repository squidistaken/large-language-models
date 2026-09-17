# Cross-Lingual Multi-Label Emotion Classification Using Large Language Models

Repository for the Large Language Models course (WBAI065-05) at the University of Groningen.

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

The project exposes a small Hydra-powered CLI, run via [main.py](main.py). Configuration defaults live in [conf/main.yaml](conf/main.yaml) and can be overridden on the command line using standard Hydra syntax (`key=value`).

Running without an `action` override launches an interactive menu where you can pick an action from a list. To run an action directly, pass it as an override:

```bash
uv run main.py action=download_data
```

### Downloading Data

Downloads the [XED](https://github.com/Helsinki-NLP/XED) annotated and projection data files for one or more languages.

Interactively, choose "Download Data" from the menu and you'll be prompted for:

- **Languages:** A comma-separated list of language codes, or `all` for every supported
  language.
- **Output directory:** Where the files are saved (default: `data/raw/xed`).
- **Force re-download:** Whether to overwrite files that already exist locally.

Non-interactively, run it directly with Hydra overrides:

```bash
uv run main.py action=download_data data.languages=[en,hr] data.out_dir=data/raw/xed data.force=false
```

Files that already exist in the output directory are skipped unless `data.force=true`.

