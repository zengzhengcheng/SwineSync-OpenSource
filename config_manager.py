import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "language": "zh",
    "batch_size": 2,
    "cache_size": 1,
    "correct": True,
    "interpolation_algorithm": "original_timestamps",
}


def normalize_config(data):
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)

    if "origin_resample" in data and "interpolation_algorithm" not in data:
        merged["interpolation_algorithm"] = (
            "original_timestamps" if data["origin_resample"] else "uniform_grid"
        )

    if merged["interpolation_algorithm"] not in {"original_timestamps", "uniform_grid"}:
        merged["interpolation_algorithm"] = DEFAULT_CONFIG["interpolation_algorithm"]

    return merged


def load_config():
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    return normalize_config(data)


def save_config(config):
    merged = normalize_config(config)
    CONFIG_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
