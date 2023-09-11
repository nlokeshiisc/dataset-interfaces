# # %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import shutil
import sys

# comment this out if you are using the pip package
sys.path.append("../")
sys.path.append("/raid/infolab/nlokesh/dataset-interfaces/")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset_interfaces import utils
from dataset_interfaces import run_textual_inversion, run_textual_inversion_plus
from dataset_interfaces import generate
import dataset_interfaces.imagenet_utils as in_utils
import dataset_interfaces.inference_utils as infer_utils
from pathlib import Path

create_confounded_dataset = True
N = 20  # N <= 20, number of class embeddings to be learnt

# %%
concatenate_dfs = []
backgrounds = ["at night", "in the fog", "in the forest", "in the rain", "in the snow"]
for bg in backgrounds:
    bg_file = "_".join(bg.split())
    df = pd.read_csv(
        f"/raid/infolab/nlokesh/dataset-interfaces/cache/{bg_file}_preds.csv"
    )
    temp_df = pd.DataFrame({"cnf": df["cnf"], "bg": bg, "z": df["true_y"]})
    concatenate_dfs.append(temp_df)
main_df = pd.concat(concatenate_dfs, ignore_index=True)

# %%
std_z_bg = main_df.groupby(["z", "bg"])["cnf"].std().reset_index()
std_z_bg = std_z_bg.sort_values(by="cnf", ascending=True)
low_std_z_bg = std_z_bg[std_z_bg["cnf"] < 0.05]

mean_z_bg = main_df.groupby(["z", "bg"])["cnf"].mean().reset_index()
mean_z_bg = mean_z_bg.sort_values(by="cnf", ascending=False)
mask = (
    mean_z_bg[["z", "bg"]]
    .apply(tuple, axis=1)
    .isin(low_std_z_bg[["z", "bg"]].apply(tuple, axis=1))
)
mean_z_bg_low_std = mean_z_bg[mask]
# # mean_z_bg_low_std[mean_z_bg_low_std['cnf'] < 0.99]
# mean_z_bg_low_std

mean_z = main_df.groupby(["z"])["cnf"].mean().reset_index()
mean_z = mean_z.sort_values(by="cnf", ascending=False)
well_understood = mean_z.head(300)
well_understood_z = well_understood["z"]

z_bg_filtered = mean_z_bg_low_std[mean_z_bg_low_std["z"].isin(well_understood_z)]
# remove z which have only one low stddev bg (so that we can choose)
z_bg_filtered = z_bg_filtered.groupby(["z"]).filter(lambda grp: len(grp) > 1)


def select_bg(group):
    return group.sample(1, random_state=0)


# now remove repeated z we select a random bg for each z
select_z_bg = (
    z_bg_filtered.groupby(["z"])[["bg", "cnf"]].apply(select_bg).sort_values(by="cnf")
)

select_z_bg = select_z_bg.reset_index().drop("level_1", axis=1)
top_z_bg = select_z_bg.head(10)
bottom_z_bg = select_z_bg.tail(10)
top_z_beta = pd.concat([top_z_bg, bottom_z_bg])
top_z_beta

# %%
from dataset_interfaces.imagenet_utils import *


def idx_to_foldername(idx):
    if idx != 0:
        return IMAGENET_IDX_TO_SYNSET[f"{idx}"]["id"]
    else:
        return "0"


def idx_to_label(idx):
    return IMAGENET_IDX_TO_SYNSET[f"{idx}"]["label"].split(",")[0]
    # return IMAGENET_COMMON_CLASS_NAMES[idx]


def spaces_to_underscores(label):
    """
    use for converting label / background strings to file names
    """
    return "_".join(label.split())


# %%
imagenet_star_path = "/raid/infolab/nlokesh/dataset-interfaces/data/imagenet_star/"
confounded_dataset_path = os.path.join(imagenet_star_path, "confounded_dataset/")
for z_idx, beta_string in zip(top_z_beta["z"], top_z_beta["bg"]):
    bg_path_local = spaces_to_underscores(beta_string)
    bg_path = os.path.join(imagenet_star_path, bg_path_local)
    foldername = idx_to_foldername(z_idx)
    img_path = os.path.join(bg_path, foldername, "00.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(bg_path, foldername, foldername, "00.jpg")

    from_address = img_path
    if not os.path.exists(
        os.path.join(confounded_dataset_path, bg_path_local, str(z_idx))
    ):
        os.makedirs(os.path.join(confounded_dataset_path, bg_path_local, str(z_idx)))
    to_address = os.path.join(
        confounded_dataset_path, bg_path_local, str(z_idx), f"00.jpg"
    )
    if create_confounded_dataset == True:
        shutil.copy(from_address, to_address)

# %%
# Now that we have constructed the training dataset with confounding between beta and z, let us try to learn embeddings z*
print("Currently hardcoded to 'confounded_dataset' folder")
IMAGENET_ROOT = (
    "/raid/infolab/nlokesh/dataset-interfaces/data/imagenet_star/confounded_dataset"
)

# %%
# a subset of ImageNet classes
classes = list(top_z_beta["z"])
betas = list(top_z_beta["bg"])
confidences = list(top_z_beta["cnf"])
class_names = [IMAGENET_COMMON_CLASS_NAMES[c] for c in classes]
tokens = [f"<{class_names[i]}-{i}>" for i in range(len(class_names))]

# %%
# path where to store an encoder, which we will load in with the learned tokens
T = 3000
encoder_root = "./encoder_root/dsi_w_high_contrast"
# torch.autograd.set_detect_anomaly(True)
embeds, cnf_error = run_textual_inversion_plus(
    IMAGENET_ROOT,
    tokens=tokens,
    z_objects=classes,
    betas=betas,
    confidences=confidences,
    z_names=class_names,
    max_train_steps=T,
    weights=[1, 1, 1],
)

infer_utils.create_encoder(
    embeds=embeds, tokens=tokens, class_names=class_names, encoder_root=encoder_root
)


# %%
def get_z_beta_path(train_path, z_idx, beta):
    path_string = os.path.join(train_path, "_".join(beta.split()), str(z_idx))
    if os.path.exists(path_string):
        return path_string
    else:
        raise RuntimeError(
            f"images for the (z, beta) pair ({z_idx}, {beta}) do not exist at the expected location {path_string} "
        )


train_data_dirs = [
    get_z_beta_path(IMAGENET_ROOT, z_idx, beta) for z_idx, beta in zip(classes, betas)
]

# %%
# path where to store an encoder, which we will load in with the learned tokens
encoder_root = "./encoder_root/vanilla_dsi_on_confounded"
embeds = []
for i in range(len(classes)):
    # runs textual inversion on a single class
    embed = run_textual_inversion(
        train_data_dirs[i],
        token=tokens[i],
        class_name=class_names[i],
    )

    embeds.append(embed)
infer_utils.create_encoder(
    embeds=embeds, tokens=tokens, class_names=class_names, encoder_root=encoder_root
)
