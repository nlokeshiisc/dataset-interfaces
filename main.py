from src import main_helper as mh
import constants as constants
import argparse

import constants as constants
from pathlib import Path
import os
from utils import common_utils as cu
from src import data_helper as dh
import pandas as pd
from src import rec_helper as rech
from utils import torch_utils as tu
from src import models
import numpy as np
from src import dataset as ds

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import argparse
import config

config = config.config

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Use the correct argument", default="config.py")
args, unknown_args = parser.parse_known_args()
config = cu.parse_args(args, unknown_args)

cu.set_logger(config)
constants.logger.info(f"Config: {cu.dict_print(config)}")
cu.set_sw(config)

# %% Parse config
trn_args = config[constants.TRAIN_ARGS]

shifts = config[constants.GENERAL_SPECS][constants.SHIFT]
shifts_idx = [constants.shift_idx[entry] for entry in shifts]

sel_classes = sorted(config[constants.DATASET_SPECS][constants.SEL_CLASSES])

rec_args = config[constants.REC_SPECS][constants.KWARGS]
rec_model_name = config[constants.REC_SPECS][constants.MODEL_NAME]

# %% Cls model
cls_model = mh.get_model(model_name="resnet50", pretrn=True)

if trn_args[constants.COMPUTE_RHO] == True:
    # Compute and save the classifier's rho on the different shifts
    for shift in shifts:
        print(f"computing and caching rho for the shift: {shift}")
        imstar_ds, dl = mh.get_ds_dl(dataset_name=shift)
        mh.evaluate_model(model=cls_model, loader=dl, cache=True)

# %% Check Accuracies
if False:
    # This is simply to sanity check the cached dataframes
    shift_ds, acc_meters = {}, {}
    for shift in shifts:
        imstar_ds, dl = mh.get_ds_dl(dataset_name=shift)
        shift_ds[shift] = imstar_ds
        acc_meter = mh.df_to_acc(dataset_name=shift)
        acc_meters[shift] = acc_meter

        print(f"Dataset: {shift}, accuracy: {acc_meters[shift].accuracy()}")
    pass

# %% Rec model

if trn_args[constants.REC] == True:
    shift_ds = None

    rec_model: models.TarnetRecModel = models.TarnetRecModel(
        datasets=shift_ds, **rec_args
    )

    df = {
        "image_files": [],
        "labels": [],
        "shifts": [],
        "rho": [],
    }

    for cls, cls_name in enumerate(sel_classes):
        for idx in range(10, 50):
            sampled_shifts = np.random.choice(shifts, 2)
            for sampled_shift in sampled_shifts:
                img_file: Path = (
                    constants.imagenet_star_dir
                    / sampled_shift
                    / cls_name
                    / f"{idx}.jpg"
                )
                assert img_file.exists()
                df["image_files"].append(img_file)
                df["labels"].append(cls)
                df["shifts"].append(sampled_shift)
                df["rho"].append(shift_ds[f"{sampled_shift}_rho"][idx])

    df = pd.DataFrame(df)

    rec_ds = ds.DfDataset(
        dataset_name=constants.IMSTAR,
        dataset_split="train",
        df=df,
        transform=constants.RESNET_TRANSFORMS[constants.TRN],
    )
    rec_dh = dh.RecDataHelper(
        dataset_name=constants.IMSTAR, dataset_type="real", trn_ds=rec_ds, **rec_args
    )

    rec_helper = rech.RecHelper(
        rec_model=rec_model,
        rec_dh=rec_dh,
        rec_model_name=rec_model_name,
    )
