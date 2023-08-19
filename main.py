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
from src import models
import numpy as np
from src import dataset as ds
from dataset_interfaces import utils as dfi_utils

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


# %% Rec model

if trn_args[constants.REC] == True:
    shifts_ds: dict = mh.get_rec_datasets(shifts=shifts)

    rec_model: models.TarnetRecModel = models.TarnetRecModel(
        datasets=shifts_ds, **rec_args
    )

    df = {
        constants.IMGFILE: [],
        constants.LABEL: [],
        constants.SHIFT: [],
        constants.RHO: [],
    }

    for cls, cls_name in enumerate(sel_classes):
        for idx in range(10, 50):
            sampled_shifts = np.random.choice(shifts, 2)
            for sampled_shift in sampled_shifts:
                shift_ds: dfi_utils.ImageNet_Star_Dataset = sampled_shifts[
                    sampled_shift
                ]["ds"]
                shift_rho: ds.DFRho = sampled_shifts[sampled_shift][constants.RHO]

                img_file: Path = (
                    constants.imagenet_star_dir
                    / sampled_shift
                    / cls_name
                    / f"{idx}.jpg"
                )
                assert img_file.exists()
                df[constants.IMGFILE].append(img_file)
                df[constants.LABEL].append(cls)
                df[constants.SHIFT].append(sampled_shift)
                df[constants.RHO].append(
                    shift_rho.get_item(image_file=img_file)[constants.RHO]
                )

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

    rec_helper.train_rec(**rec_args)
    rec_helper.save_model()
