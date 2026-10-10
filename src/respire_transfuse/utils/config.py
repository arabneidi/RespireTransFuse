from pathlib import Path
import yaml


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(paths_yaml, experiment_yaml):
    paths_cfg = load_yaml(paths_yaml)
    exp_cfg = load_yaml(experiment_yaml)
    return deep_merge(paths_cfg, exp_cfg)


def save_yaml(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
