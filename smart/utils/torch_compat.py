import torch
from easydict import EasyDict


def register_checkpoint_safe_globals() -> None:
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is not None:
        add_safe_globals([EasyDict])


def torch_load_compat(path, map_location=None, weights_only=None):
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location

    if weights_only is not None:
        try:
            return torch.load(path, weights_only=weights_only, **load_kwargs)
        except TypeError:
            pass

    return torch.load(path, **load_kwargs)
