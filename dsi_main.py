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
print("Currently hardcoded to base folder")
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
classes = [236, 651, 10]
class_names = [in_utils.IMAGENET_COMMON_CLASS_NAMES[c] for c in classes]
tokens = [f"<{class_names[i]}-{i}>" for i in range(len(class_names))]

# train_data_dirs = [os.path.join(IMAGENET_ROOT, "train", in_utils.IMAGENET_IDX_TO_SYNSET[str(c)]['id']) for c in classes]
train_data_dirs = [
    os.path.join(IMAGENET_ROOT, in_utils.IMAGENET_IDX_TO_SYNSET[str(c)]["id"])
    for c in classes
]


# %%
print(train_data_dirs)

# %% [markdown]
# #### Run textual inversion

# %%
# embeds = []
# for i in range(len(classes)):
#     # runs textual inversion on a single class
#     embed = run_textual_inversion(
#         train_data_dirs[i], token=tokens[i], class_name=class_names[i]
#     )

#     embeds.append(embed)

# %% [markdown]
# #### Add to tokenizer and text encoder

# %%
# infer_utils.creat_encoder(
#     embeds=embeds, tokens=tokens, class_names=class_names, encoder_root=encoder_root
# )

# %% [markdown]
# ### Option 2: Create Encoder Root for ImageNet
# To use our learned tokens for the ImageNet (ImageNet*), we save a tokenizer and text encoder with the 1k tokens. This could take 6+ minutes

# %%
# Get the tokens from huggingface
os.system(
    "wget https://huggingface.co/datasets/madrylab/imagenet-star-tokens/resolve/main/tokens.zip"
)
os.system("unzip tokens.zip")

# %%
token_path = "./tokens"
infer_utils.create_imagenet_star_encoder(
    token_path, encoder_root="./encoder_root_imagenet"
)

# %% [markdown]
# ### Generate Counterfactual Examples
#
# To use our learned class tokens for the ImageNet Dataset (ImageNet*), keep `use_provided=True` <br>
# To use the tokens learned in the cells above, set `use_provided=False`

# %%
use_provided = True

# %%
if use_provided:
    classes = [236, 651, 10]
    class_names = [in_utils.IMAGENET_COMMON_CLASS_NAMES[c] for c in classes]
    root = "./encoder_root_imagenet"


else:
    classes = [0, 1, 2]
    class_names = class_names
    root = encoder_root

# %% [markdown]
# #### A small set of distribution shifts, as examples

# %%
shifts = ["base", "in the grass", "in the snow", "in bright sunlight", "oil painting"]
prompts = [
    "a photo of a <TOKEN>",
    "a photo of a <TOKEN> in the grass",
    "a photo of a <TOKEN> in the snow",
    "a photo of a <TOKEN> in bright sunlight",
    "an oil painting of a <TOKEN>",
]

# %% [markdown]
# #### Generating counterfactual examples in the shifts above for each class

# %%
imgs = []
seed = 0
for c in classes:
    imgs_class = generate(
        root, c, prompts, num_samples=1, random_seed=range(seed, seed + len(prompts))
    )
    imgs_class = generate(
        root, c, prompts, num_samples=1, random_seed=range(seed, seed + len(prompts))
    )
    imgs_class = [imgs[0] for imgs in imgs_class]
    seed += len(prompts)

    imgs.append(imgs_class)

# %%
utils.visualize_samples(imgs, class_names, shifts, dpi=200, figsize=(6, 4), fontsize=8)

# %% [markdown]
# ### CLIP Metrics
# To directly evaluate the quality of the generated image, we use CLIP similarity to quantify the presence of the object of interest and desired distribution shift in the image.

# %% [markdown]
# We can measure the extent to which generated images for the class "doberman" contain a doberman as follows:

# %%
infer_utils.clip_similarity(imgs[0], "a photo of a doberman")

# %% [markdown]
# We can measure the extent to which generated images in the grass are indeed in the grass as follows:

# %%
infer_utils.clip_similarity([imgs[i][1] for i in range(3)], "a photo in the grass")
