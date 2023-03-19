import importlib
from typing import Union

from omegaconf import OmegaConf, DictConfig


def instantiate_from_config(config: Union[str, DictConfig], **extra_kwargs):
    if isinstance(config, str):
        return dict({
            k: instantiate_from_config(v)
            for k, v in OmegaConf.load(config).items()})
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if isinstance(config, DictConfig):
        config = OmegaConf.to_object(config)
    params = config.get("params", dict())
    extra_kwargs.update(params)
    return get_obj_from_str(config["target"])(**extra_kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
