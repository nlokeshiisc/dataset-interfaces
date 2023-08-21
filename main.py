from src import main_helper as mh
import constants as constants
import argparse

import constants as constants
from pathlib import Path
import os
from utils import common_utils as cu
import pandas as pd
from src import models
import numpy as np
from src import dataset as ds
from dataset_interfaces import utils as dfi_utils
import copy

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

    trn_df = {
    trn_df = {
    trn_df = {
        constants.IMGFILE: [],
        constants.LABEL: [],
        constants.SHIFT: [],
        constants.RHO: [],
        constants.LOSS: [],
        constants.LOSS: [],
    }
    tst_df = copy.deepcopy(trn_df)

    for cls, cls_name in enumerate(sel_classes):
        for idx in range(10, 50):
            sampled_shifts = np.random.choice(shifts, 2)
            for ss in shifts:
                shift_ds: dfi_utils.ImageNet_Star_Dataset = shifts_ds[ss]["ds"]
                cache_ds: ds.DFRho = shifts_ds[ss]["cache"]
                img_file: Path = (
                    constants.imagenet_star_dir / ss / cls_name / f"{idx}.jpg"
                )
                assert img_file.exists(), "The image file does not exist"
                rho_loss = cache_ds.get_item(image_file=img_file)
                shift_rho = rho_loss[constants.CNF]
                shift_loss = rho_loss[constants.LOSS]

                tst_df[constants.IMGFILE].append(img_file)
                tst_df[constants.LABEL].append(cls)
                tst_df[constants.SHIFT].append(ss)
                tst_df[constants.RHO].append(shift_rho)
                tst_df[constants.LOSS].append(shift_loss)

                if ss in sampled_shifts:
                    trn_df[constants.IMGFILE].append(img_file)
                    trn_df[constants.LABEL].append(cls)
                    trn_df[constants.SHIFT].append(ss)
                    trn_df[constants.RHO].append(shift_rho)
                    trn_df[constants.LOSS].append(shift_loss)

    trn_df = pd.DataFrame(trn_df)
    tst_df = pd.DataFrame(tst_df)

    rec_dh = mh.get_rec_dh(
        trn_df=trn_df,
        tst_df=tst_df,
        **rec_args,
    )

    rec_helper = rech.RecHelper(
        rec_model=rec_model,
        rec_dh=rec_dh,
        rec_model_name=rec_model_name,
        **rec_args,
    )

    # %% Train the rec model
    rec_helper.train_rec(**rec_args)
    rec_helper.save_model()
