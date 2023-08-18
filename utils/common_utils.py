import random
import torch
import numpy as np
from copy import deepcopy
import json
import constants
from pathlib import Path
import importlib.util
import os
import logging
from tensorboardX import SummaryWriter
import shutil


def set_seed(seed: int = 42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cuda_device(gpu_num):
    import os

    # Set the environmental variable to discard all other GPUs
    if type(gpu_num) == int:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    else:
        raise ValueError(f"Only int supported. {gpu_num} not supported")


def get_device():
    return "cuda:0"


def do_post():
    if constants.sw is not None:
        constants.sw.close()
    if constants.logger is not None:
        constants.logger.handlers = []


def get_log_dir(config: dict = None, sw=False) -> Path:
    """Sets the logger in constants

    Args:
        config (dict, optional): _description_. Defaults to None.
    """
    if not sw:
        log_dir: Path = constants.LOG_DIR
    else:
        log_dir: Path = constants.TBDIR

    dataset_name = config[constants.DATASET_SPECS][constants.DATASET_NAME]
    log_dir = log_dir / dataset_name

    return log_dir


def set_sw(config: dict, summ_dir=None):
    """Sets the summarywriter in constants.sw"""
    if summ_dir is None:
        summ_dir = get_log_dir(config, sw=True)
    summ_dir: Path = summ_dir / config[constants.EXPTNAME]
    if os.path.exists(str(summ_dir.absolute())):
        shutil.rmtree(
            str(summ_dir.absolute()), ignore_errors=True
        )  # ignore dir not empty error
    constants.sw = SummaryWriter(str(summ_dir.absolute()))
    constants.logger.info(f"Writing tensorboard logs to {summ_dir}")


def set_logger(config: dict = None, log_dir: Path = None, log_file_name: str = None):
    # Also set the logger here

    log_file_mode = config.get(constants.LOGMODE, "w")

    if log_dir is None:
        log_dir = get_log_dir(config)
    if log_file_name is None:
        log_file_name = f"{config[constants.EXPTNAME]}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file_name[-4:] == ".log":
        log_file_name = log_file_name[:-4]
    logfile_path = str((log_dir / f"{log_file_name}.log").absolute())
    print("Writing logs to: ", logfile_path)

    logging.basicConfig(
        filename=logfile_path,
        format="%(asctime)s :: %(levelname)s :: %(filename)s:%(funcName)s :: %(message)s",
        filemode=log_file_mode,
    )
    logger = logging.getLogger(name="simulator_recourse_2")
    logger.setLevel(logging.DEBUG)
    constants.logger = logger


def parse_args(args, unknown_args) -> dict:
    """This API is a facade for all the code modulated changes that are to be made to the config.
    The following rules are followed that automatically applies changes to the config:
        1. If rnd sim is used, then we add suffix -rnd=sim to
            ver
            rec
            exptname
        2. We add -imgemb=True/False to the ver model name
        3. We add -ver=accrej/diffdiff to:
            ver -- only if rnd_sim is False
            rec -- only if use_ver is True
            exptname -- only when rec_trn=False or use_ver=True
        @Note: We will do (3) only if sim is not random
        4. We suffix the dataset corruption ratio to the exptname
        5. We push the sim_args to kwargs of the classifier


    Args:
        args (_type_): _description_
        unknown_args (_type_): _description_

    Returns:
        dict: _description_
    """
    config = __load_config_file(args, unknown_args)

    dict_print(config)

    gpuid = config[constants.GPUID]
    set_cuda_device(gpuid)
    seed = config[constants.SEED]
    constants.seed = seed
    set_seed(seed)

    return config


def __load_config_file(args, unknown_args):
    """This method is private to this file. Do not call from outside!

    Args:
        args (_type_): _description_
        unknown_args (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_name = os.path.splitext(args.config)[0]
    file_name = file_name.split("/")[0]
    spec = importlib.util.spec_from_file_location(file_name, args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    for override_config in unknown_args:
        parts = override_config.split(":")
        key = parts[0]
        value = parts[1]

        if "." in key:
            key_parts = key.split(".")
            primary_key = key_parts[0]
            secondary_key = key_parts[1]
            config[primary_key][secondary_key] = evaluate_value(value)
        else:
            config[key] = evaluate_value(value)
    return config


def evaluate_value(value):
    if "[" in value:
        # processing list arguments
        values = value.replace("[", "").replace("]", "")
        values = values.split(",")
        try:
            values = [eval(entry) for entry in values]
        except:
            pass
        return values
    else:
        try:
            return eval(value)
        except:
            return value


def insert_kwargs(kwargs: dict, new_args: dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args


def dict_print(d: dict):
    d_new = deepcopy(d)

    def cast_str(d_new: dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new

    d_new = cast_str(d_new)

    pretty_str = json.dumps(d_new, sort_keys=False, indent=4)
    print(pretty_str)
    return pretty_str


def get_dirs(path: Path) -> list:
    """Finds the sub directories in the path and returns them

    Args:
        path (Path): _description_

    Returns:
        list: _description_
    """
    dirs = list(path.glob("*/"))
    return dirs


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def sample_rand_dict(sample_dict: dict):
    """Returns a random value from the dictionary

    Args:
        dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.random.choice(list(sample_dict.values()))


def match_row(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Returns the indices of rows in the array that match the row

    Args:
        arr (np.ndarray): _description_
        row (np.ndarray): _description_
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    if isinstance(row, torch.Tensor):
        row = row.numpy()
    return np.where((arr == row).all(axis=1))[0]