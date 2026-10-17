"""Reading the YAML configuration files."""
from pathlib import Path
from types import SimpleNamespace

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _merge(base: dict, override: dict) -> dict:
    """Merges two nested dictionaries; values from override win."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_namespace(d):
    """Turns nested dictionaries into objects with attributes (cfg.mesh.Nr)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


def load_config(path):
    """Reads base.yaml, applies the case YAML on top and returns (cfg, dict)."""
    path = Path(path)
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        case = yaml.safe_load(f) or {}
    merged = _merge(base, case)
    return _to_namespace(merged), merged
