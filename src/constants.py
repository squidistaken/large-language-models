# Paths.
RAW_DIR = "data/raw/xed"
PROCESSED_DIR = "data/processed/xed"

# Reproducibility.
DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "dev", "test")
TRAIN, DEV, TEST = SPLIT_NAMES

# The partitions a model is scored on.
EVAL_SPLITS = (DEV, TEST)
DEFAULT_SPLITS = (0.70, 0.20, 0.10)

# The eight permitted labels, in the fixed order used by every downstream metric.
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")

# Where a pair's English labels came from.
GOLD = "gold"
PROJECTED = "projected"

# Decided languages.
SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "hr"

# XED stores many English annotations by Danish subtitle id, so Danish ids are followed during
# alignment to find the English line, and never appear in the splits.
PIVOT_LANG = "da"

# The languages XED annotated by hand.
ANNOTATED_LANGS = {"en", "fi"}

# The five resource bands from the course project.
LANGUAGE_BANDS: dict[int, set[str]] = {
    1: {"mk", "vi", "sk", "is"},
    2: {"da", "ru", "bs", "sl", "ar", "no"},
    3: {"he", "sv", "nl", "de", "hu", "hr"},
    4: {"cs", "it", "bg", "pl", "pt", "fr"},
    5: {"el", "fi", "sr", "tr", "ro", "es", "pt_br"},
}
BAND_NAMES = {1: "extreme low", 2: "very low", 3: "mid-low", 4: "mid-high", 5: "high"}
BAND_SIZES = {
    1: "under ~1,000",
    2: "~1,000-4,400",
    3: "~4,400-6,000",
    4: "~6,000-7,300",
    5: "~8,000 and above",
}
LANGUAGE_TO_BAND = {lang: band for band, langs in LANGUAGE_BANDS.items() for lang in langs}
ALL_LANGUAGES = sorted(set(LANGUAGE_TO_BAND) | {SOURCE_LANG})
TARGET_LANGS = sorted(LANGUAGE_TO_BAND)
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bs": "Bosnian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "mk": "Macedonian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt_br": "Portuguese (Brazil)",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

# The systems compared: zero-shot prompting and the PEFT conditions (English, the target language,
# or both).
ZERO_SHOT = "zero_shot"
BILINGUAL = "bilingual"

# Number of demonstrations in the five-shot prompt
FIVE_SHOT = 5

# The run folders.
ADAPTER_DIR = "adapter"
CHECKPOINT_DIR = "checkpoints"
RUN_INFO_FILE = "run_info.json"
SCORES_DIR = "scores"
SELECTION_FILE = "selection.json"

# The metrics.
PRIMARY_METRICS = ("micro_f1", "jaccard_samples")
SECONDARY_METRICS = ("macro_f1", "micro_precision", "micro_recall", "hamming_loss")
DIAGNOSTIC_METRICS = ("exact_match", "malformed_rate", "empty_rate")
AGGREGATE_METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS, *DIAGNOSTIC_METRICS)

SHORT_NAMES = {
    "micro_f1": "micro-F1",
    "jaccard_samples": "Jaccard",
    "macro_f1": "macro-F1",
    "micro_precision": "micro-P",
    "micro_recall": "micro-R",
    "hamming_loss": "Hamming",
    "exact_match": "Exact",
    "malformed_rate": "Malformed",
    "empty_rate": "Empty",
}

LOWER_IS_BETTER = {"hamming_loss", "malformed_rate", "empty_rate"}

# The metrics with bootstrap confidence intervals.
HEADLINE_METRICS = ("micro_f1", "jaccard_samples", "macro_f1")

# The dev quantities that can pick the kept checkpoint, and whether higher is better.
CHECKPOINT_METRICS = {"loss": False, "micro_f1": True}
