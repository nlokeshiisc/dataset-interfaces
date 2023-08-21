import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import constants as constants
import numpy as np
from joblib import Parallel, delayed

GPU = 1

shifts = [
    # constants.BASE,
    # constants.DUSK,
    constants.FOREST,
    # constants.FOG,
    # constants.NIGHT,
    # constants.SNOW,
    constants.RAIN,
    # constants.STUDIO,
    constants.SUNLIGHT,
    constants.RAIN,
]


def run_shift(shift):
    cmd = f"python main.py \
        {constants.GENERAL_SPECS}.{constants.SHIFT}:[{shift}] \
        {constants.GPUID}:{GPU}"
    print(cmd)
    os.system(cmd)


Parallel(n_jobs=3)(delayed(run_shift)(shift) for shift in shifts)
