from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


def apply_fold(cfg, fold_name):
    """Set train/val years from split.folds[fold_name] and put outputs in <output_dir>/<fold_name>.

    Works on the raw config dict (before dict_to_namespace); used by train.py and evaluate.py so
    both resolve a fold, and so its run directory, the same way.
    """
    fold = cfg["split"]["folds"][fold_name]     # KeyError = unknown fold name
    cfg["split"]["train"]["years"] = fold["train"]
    cfg["split"]["val"]["years"] = fold["val"]
    cfg["output_dir"] = f'{cfg["output_dir"]}/{fold_name}'
    return cfg
