# %%
import os
import sys

# comment this out if you are using the pip package
sys.path.append("../")

import torch
import matplotlib.pyplot as plt
from dataset_interfaces import utils
from dataset_interfaces import run_textual_inversion
from dataset_interfaces import generate
import dataset_interfaces.imagenet_utils as in_utils
import dataset_interfaces.inference_utils as infer_utils
from pathlib import Path

# set root to ImageNet dataset
IMAGENET_ROOT = "/raid/infolab/nlokesh/dataset-interfaces/data/imagenet_star/base"

# %% [markdown]
# To easily use the learned tokens in text prompts, we load the learned token-embedding pairs into a tokenizer and corresponding text encoder. Below we define the path where we will store the tokenizer and encoder.

# %%
# path where to store an encoder, which we will load in with the learned tokens
encoder_root = "./encoder_root"

# %% [markdown]
# ### Option 1: Construct Dataset Interface

# %%
# a subset of ImageNet classes
classes = [7, 16, 26, 31, 49, 56, 71, 84, 105, 113]
class_names = [in_utils.IMAGENET_COMMON_CLASS_NAMES[c] for c in classes]
tokens = [f"<{class_names[i]}-{i}>" for i in range(len(class_names))]

# train_data_dirs = [os.path.join(IMAGENET_ROOT, "train", in_utils.IMAGENET_IDX_TO_SYNSET[str(c)]['id']) for c in classes]
train_data_dirs = [
    os.path.join(IMAGENET_ROOT, in_utils.IMAGENET_IDX_TO_SYNSET[str(c)]["id"])
    for c in classes
]


# %%
train_data_dirs

# %%
tokens

# %% [markdown]
# #### Run textual inversion

# %%
embeds = {}
for i in range(len(classes)):
    # runs textual inversion on a single class
    embed = run_textual_inversion(
        train_data_dirs[i], token=tokens[i], class_name=class_names[i]
    )

    embeds[classes[i]] = embed

    import pickle

    with open("base_sstar_embed.pkl", "wb") as f:
        pickle.dump(embeds, f)
