from importlib import import_module

from experiments.aesthetic_continuous.config import TrainConfig as AestheticContinuousTrainConfig


def _optional_train_config(module_name):
    try:
        return import_module(module_name).TrainConfig
    except ModuleNotFoundError:
        return None


CONFIG_MAPPING = {
                "aesthetic_continuous": AestheticContinuousTrainConfig,
               }

OPTIONAL_CONFIGS = {
    "ram_insertion": "experiments.ram_insertion.config",
    "usb_pickup_insertion": "experiments.usb_pickup_insertion.config",
    "object_handover": "experiments.object_handover.config",
    "egg_flip": "experiments.egg_flip.config",
}

for exp_name, module_name in OPTIONAL_CONFIGS.items():
    train_config = _optional_train_config(module_name)
    if train_config is not None:
        CONFIG_MAPPING[exp_name] = train_config
