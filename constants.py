from pathlib import Path
from torchvision import transforms
import numpy as np
from logging import Logger
from tensorboardX import SummaryWriter
import logging


PROJ_DIR = Path(".").absolute()
IMSTAR = "imagenet_star"

# %% Imagenet constants
imagenet_star_dir: Path = PROJ_DIR / "data/imagenet_star"

BASE = "base"
DUSK = "at_dusk"
NIGHT = "at_night"
SUNLIGHT = "in_bright_sunlight"
FOG = "in_the_fog"
FOREST = "in_the_forest"
RAIN = "in_the_rain"
SNOW = "in_the_snow"
STUDIO = "studio_lighting"

# %% Dataset constants
TRN = "trn"
VAL = "val"
TST = "tst"
TRNTST = "trntst"

# %% model constants

RESNET_TRANSFORMS = {
    TRN: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    TST: transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
RESNET_TRANSFORMS[VAL] = RESNET_TRANSFORMS[TST]
RESNET_TRANSFORMS[TRNTST] = RESNET_TRANSFORMS[TST]

shift_idx: dict = {
    BASE: 0,
    DUSK: 1,
    NIGHT: 2,
    SUNLIGHT: 3,
    FOG: 4,
    FOREST: 5,
    RAIN: 6,
    SNOW: 7,
    STUDIO: 8,
}

SHIFT = "shift"


RANDOM_SIM = "random_simulator"

logger: Logger = None
seed: int = 0
sw: SummaryWriter = None

CLASSID = "class_id"
SEL_CLASSES = "sel_classes"
GPUID = "gpu_id"
SEED = "seed"
LOG_DIR = Path("results/logs")
TBDIR = Path("results/tb")

betatostr_fn = lambda x: "".join([str(i) for i in x])

CORRUPTIONS = [
    "gaussian_noise",
    "impulse_noise",
    "defocus_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
]


EXPTNAME = "expt_name"
GENERAL_SPECS = "general_specs"
LOGMODE = "log_mode"
DATASET_SPECS = "dataset_specs"
CLS_SPECS = "cls_specs"
VERIFIER_SPECS = "verifier_specs"
ZBETA_SPECS = "zbeta_specs"
TRN_Z = "trn_z"

EMB_SPECS = "emb_specs"
REC_SPECS = "rec_specs"

VIEW = "view"
ZOOM = "zoom"
LGT_COLOR = "lgt_color"
ILLUMINATION = "illumination"
CONTRAST = "contrast"

DFPATH = "df_path"
REALDS = "real_ds"
SIMDS = "sim_ds"
SIM_PROBA = "cls_sim_probs"
IDX = "idx"
IDEALRATIO = "ideal_ratio"
CORRUPTIONRATIO = "corruption_ratio"
SEVERITY = "severity"

BATCH_SIZE = "batch_size"
NUM_WORKERS = "num_workers"
KWARGS = "kwargs"
DATASET_NAME = "dataset_name"
DATASET_SPLIT = "dataset_split"
MODEL_TYPE = "model_type"
MODEL_NAME = "model_name"
REALARGS = "real_args"
SIMARGS = "sim_args"
REAL_PREDS = "real_preds"

LABELS = "labels"
IMAGE_ID = "image_id"
LOAD_MODEL = "load_model"

INPUT = "input"
PREDZ = "pred_z"
PREDBETA = "pred_beta"
PREDY = "pred_y"
GOLDZ = "gold_z"
GOLDBETA = "gold_beta"
GOLDY = "gold_y"

EPOCHS = "epochs"
LRN_RATE = "lrn_rate"
CHECKPOINTS = "checkpoint"
MISC = "misc"
MARGIN = "margin"
CTR_LAMBDA = "ctr_lambda"
LOG_TEST_METRICS = "log_test_metrics"

TARNET_RECOURSE = "tarnet_rec"
EMBEDDING_RECOURSE = "emb_rec"
ETA_RECOURSE = "eta_rec"
TRAIN_ARGS = "train_args"
COMPUTE_RHO = "compute_rho"
CLS = "cls"
ZBETA = "zbeta"
REC = "rec"
VER = "ver"
EMB = "emb"
EMBDIM = "emb_dim"
IMGEMBDIM = "imgemb_dim"
NN_ARCH = "nn_arch"
DELTA = "delta"
IMGEMB = "imgemb"
BETA = "beta"
BETAID = "beta_id"
RHO = "rho"
GRP_RHO = "grp_rho"
FINE_GRP_RHO = "fine_grp_rho"
FINE_LABELS = "fine_labels"
Z_OUT = "z_model_output"
BATCH_NORM = "batch_norm"
SAMPLER = "sampler"
NNARCH = "nn_arch"
RESNET_CLS = "resnet_cls"
RESNET_Z = "resnet_z"

USE_IMGEMB = "use_imgembedding"
USE_VER = "use_ver"
ENFORCE_CTRLOSS = "enforce_counterfactual_loss"
VER_OUTPUT = "ver_output"
ACCREJ = "accept_reject"
DIFF_OF_DIFF = "diff_of_diffs"
DIFF_DIFF_THRES = "diff_diff_threshold"
TABLE_LOOKUP = "table_lookup"

LOAD_CLS_TRN = "load_cls_trn"
