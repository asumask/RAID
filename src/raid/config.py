from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


def apply_fold(cfg, fold_name):
    """Set split.train/split.val from split.folds[fold_name] and put outputs in <output_dir>/<fold_name>.

    The configs only define split.test, split.folds and split.full, so train/val years always come
    from a fold. fold_name "full" trains on split.full.train with no val years (final model).

    Works on the raw config dict (before dict_to_namespace); used by train.py and evaluate.py so
    both resolve a fold, and so its run directory, the same way.
    """
    if fold_name == "full":
        train, val = cfg["split"]["full"]["train"], []
    else:
        fold = cfg["split"]["folds"][fold_name]     # KeyError = unknown fold name
        train, val = fold["train"], fold["val"]
    cfg["split"]["train"] = {"years": train}
    cfg["split"]["val"] = {"years": val}
    cfg["output_dir"] = f'{cfg["output_dir"]}/{fold_name}'
    return cfg
