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
PAIRS_LANGS = set(ALL_LANGUAGES)
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

# Models.
DEFAULT_DTYPE = "bfloat16"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 32
QUANTIZATION_4BIT = "4bit"
