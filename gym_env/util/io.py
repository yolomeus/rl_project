import os

from hydra.experimental import initialize, compose, initialize_config_dir
from hydra.utils import to_absolute_path


def load_hydra_config(file_path):
    """Load a hydra composable config or a single yaml config

    :param file_path: path to the hydra main default_config
    :return: a DictConfig containing the configuration
    """
    file_path = to_absolute_path(file_path)
    conf_dir, file_name = os.path.split(file_path)
    with initialize_config_dir(conf_dir):
        cfg = compose(config_name=file_name)
    return cfg
