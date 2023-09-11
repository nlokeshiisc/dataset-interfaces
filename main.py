# We must set cuda before any other imports.
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import argparse
import config
from utils import common_utils as cu
from dataset_interfaces.imagenet_utils import *

config = config.config

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Use the correct argument", default="/raid/infolab/nlokesh/dataset-interfaces/config.py")
args, unknown_args = parser.parse_known_args()
config = cu.parse_args(args, unknown_args)

from src import main_helper as mh
import argparse

import constants as constants
from src import models
from src import rec_helper as rech

cu.set_logger(config)
constants.logger.info(f"Config: {cu.dict_print(config)}")
cu.set_sw(config)

# %% Parse config
trn_args = config[constants.TRAIN_ARGS]

shifts = config[constants.GENERAL_SPECS][constants.SHIFT]
shifts_idx = [constants.shift_idx[entry] for entry in shifts]

sel_sysnets = config[constants.DATASET_SPECS][constants.SEL_SYSNETS]

rec_args = config[constants.REC_SPECS][constants.KWARGS]
rec_model_name = config[constants.REC_SPECS][constants.MODEL_NAME]

def idx_to_id(id):
    if idx != 0:
        return IMAGENET_IDX_TO_SYNSET[f'{idx}']['id']
    else:
        return '0'

object_z_id_list = [idx_to_id(idx) for idx in range(1000)]

# %% Cls model
cls_model = mh.get_model(model_name="resnet50", pretrn=True)

if trn_args[constants.COMPUTE_RHO] == True:
    # Compute and save the classifier's rho on the different shifts
    for shift in shifts:
        for object_z_id in object_z_id_list:
            print(f"computing and caching rho for the shift: {shift}")
            imstar_ds, dl = mh.get_ds_dl(dataset_name=shift, object_z_id=object_z_id)
            mh.evaluate_model(model=cls_model, loader=dl, cache=True)


# # %% Rec model


# shifts_ds: dict = mh.get_rec_datasets(shifts=shifts)

# rec_model: models.TarnetRecModel = models.TarnetRecModel(datasets=shifts_ds, **rec_args)

# trn_df, tst_df = mh.filter_trn_tst_df(
#     shifts=shifts, sel_sysnets=sel_sysnets, shifts_ds=shifts_ds
# )

# rec_dh = mh.get_rec_dh(
#     trn_df=trn_df,
#     tst_df=tst_df,
#     **rec_args,
# )

# rec_helper = rech.RecHelper(
#     rec_model=rec_model,
#     cls_model=cls_model,
#     rec_dh=rec_dh,
#     rec_model_name=rec_model_name,
#     **rec_args,
# )

# if trn_args[constants.REC] == True:
#     # %% Train the rec model
#     rec_helper.train_rec(**rec_args)
#     rec_helper.save_model()

# else:
#     rec_helper.load_model()
#     rec_helper.evaluate_rec(save_probs=False, dataset_split=constants.TST)
