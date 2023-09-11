import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import pickle

import wandb

wandb.login()


import PIL
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from dataset_interfaces import utils
from dataset_interfaces.templates import (
    imagenet_templates_small,
    imagenet_style_templates_small,
)
from dataset_interfaces import generate
import dataset_interfaces.inference_utils as infer_utils

from copy import deepcopy

logger = get_logger(__name__)


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]
        self.num_images = len(self.image_paths)

        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        path = self.image_paths[i % self.num_images]
        image = Image.open(path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example


class TextualInversionPlusDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        background_text,
        all_background_texts,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.background_text = background_text
        self.all_background_texts = all_background_texts
        self.true_beta_idx = self.all_background_texts.index(self.background_text)

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]
        self.num_images = len(self.image_paths)

        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        """
        returns a dictionary with
        'pixel_values'
        'text_prompts'
        'true_beta_index'
        'input_ids_list'
        'object_text'
        'background_text'
        """
        example = {}
        texts = []
        example["input_ids_list"] = []
        placeholder_string = self.placeholder_token
        base_prompt = random.choice(self.templates)
        base_prompt = base_prompt.format(placeholder_string)
        for idx, background_text in enumerate(self.all_background_texts):
            texts.append(base_prompt + f" {background_text}")
            example["input_ids_list"].append(
                self.tokenizer(
                    texts[idx],
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids[0]
            )

        example["text_prompts"] = texts
        example["true_beta_index"] = self.true_beta_idx

        path = self.image_paths[i % self.num_images]
        image = Image.open(path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["object_text"] = placeholder_string
        example["background_text"] = self.background_text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
            ]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def run_textual_inversion(
    train_path,
    token,
    class_name,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
    max_train_steps=3000,
    learning_rate=5.0e-04,
):
    # Fixed Parameters
    learnable_property = "object"
    scale_lr = False
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    repeats = 1
    # revision = "fp16"
    mixed_precision = "fp16"
    tokenizer_name = None
    resolution = 768
    center_crop = True
    train_batch_size = 1
    gradient_accumulation_steps = 1
    local_rank = -1
    pin_mags = 1
    save_intermediates = True
    logging_dir = "logs"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        # log_with="tensorboard",
        # logging_dir=logging_dir,
    )

    # Load the tfokenizer and add the placeholder token as a additional special token
    if tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    elif pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    initializer_token_standin = token + "_init"
    # Convert the initializer_token, placeholder_token to ids
    initializer_token_id = utils.load_initializer_text(
        text_encoder, tokenizer, class_name, initializer_token_standin
    )
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    placeholder_token_id = tokenizer.convert_tokens_to_ids(token)
    # Load models and create wrapper for stable diffusion
    curr_emb = text_encoder.text_model.embeddings.token_embedding
    num_orig_embs = curr_emb.num_embeddings
    emb_dim = curr_emb.embedding_dim
    assert placeholder_token_id == num_orig_embs  # we only added one more. TODO change

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # convert to split embedding
    if pin_mags == 1:
        mag_targets = torch.tensor([initializer_token_id])
    else:
        mag_targets = None
    new_emb = utils.SplitEmbedding(
        num_orig_embs, 1, emb_dim, magnitude_targets=mag_targets
    )
    new_emb.initialize_from_embedding(
        text_encoder.text_model.embeddings.token_embedding
    )
    text_encoder.text_model.embeddings.token_embedding = new_emb

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
        text_encoder.text_model.embeddings.token_embedding.main_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    train_dataset = TextualInversionDataset(
        data_root=train_path,
        tokenizer=tokenizer,
        size=resolution,
        placeholder_token=token,
        repeats=repeats,
        learnable_property=learnable_property,
        center_crop=center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae and unet to device
    vae.to(accelerator.device, torch.float16)
    unet.to(accelerator.device, torch.float16)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion")
    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    wandb.init(
        project="Textual Inversion Plus", config={"method": "vanilla", "z": class_name}
    )
    for epoch in tqdm(range(num_train_epochs)):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                if revision == "fp16":
                    batch["pixel_values"] = batch["pixel_values"].type(torch.float16)

                # Convert images to latent space
                latents = (
                    vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                )
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device, torch.float16)

                if revision == "fp16":
                    noise = noise.type(torch.float16)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                if revision == "fp16":
                    encoder_hidden_states = encoder_hidden_states.type(torch.float16)

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = (
                    F.mse_loss(model_pred, target, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()
        wandb.log({"loss": loss.detach().item()})

    emb_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    learned_embeds = emb_weight[placeholder_token_id].detach().cpu()

    accelerator.end_training()

    return learned_embeds


def clip_contrastive_loss(prompts, pipe, tokenizer, text_encoder, random_seed):
    """
    returns contrastive loss between stable-diffusion image clip embeddings and prompt clip embeddings
    prompts: list of strings with appropriate tokens substituted
    """

    clip_model, clip_processor = infer_utils.load_clip_model()
    clip_model = clip_model.eval().cuda()

    cmd_args = {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
    cmd_args["generator"] = torch.Generator("cuda").manual_seed(random_seed)

    # generate images for all prompts in array while passing gradients for z*
    results = pipe(prompts, **cmd_args)
    images = results.images  # array of PIL images
    images = [image.resize((512, 512)) for image in images]

    # compute embeddings of images and prompts while passing gradients for z* in both
    text_embeds = alt_text_forward(clip_model, clip_processor, prompts=prompts)
    imgs_processed = torch.stack([clip_processor(img) for img in images]).cuda()
    image_embeds = alt_img_forward(clip_model, clip_processor, imgs=imgs_processed)

    text_embeds = text_embeds.float()
    inner_products = torch.matmul(text_embeds, image_embeds.t())
    # x -> 1 - x , for non-diagonal x
    flip_non_diags = (
        -1
        * (
            2 * torch.eye(n=inner_products.shape[0], device=inner_products.device)
            - torch.ones_like(inner_products)
        )
        * inner_products
    )

    loss = flip_non_diags.sum()

    return loss


def alt_text_forward(clip_model, clip_processor, prompts=None, inp=None, v2=True):
    # this is a modified version of inference_utils copy with autograd on for CLIP
    if v2:
        inputs = clip_processor.tokenizer(
            prompts, padding="max_length", return_tensors="pt"
        )
        text = inputs["input_ids"].cuda()

        with torch.autocast(device_type="cuda"):
            x = clip_model.token_embedding(text)

            x = x + clip_model.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = clip_model.ln_final(x)

            if clip_processor.eos_token_id is not None:
                text_embeds = (
                    x[
                        torch.arange(x.shape[0]),
                        (text == clip_processor.eos_token_id).int().argmax(dim=-1),
                    ]
                    @ clip_model.text_projection
                )

            else:
                text_embeds = (
                    x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                    @ clip_model.text_projection
                )

            text_embeds_norm = text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds_norm
            return text_embeds

    else:
        if inp is None:
            inp = clip_processor(
                text=prompts, images=None, return_tensors="pt", padding=True
            )
        else:
            assert prompts is None and imgs is None

        eos_token = clip_processor.tokenizer.eos_token
        eos_token_id = clip_processor.tokenizer.encoder[eos_token]
        inp = {k: v.cuda() for k, v in inp.items() if v is not None}
        input_ids = inp["input_ids"]
        output_attentions = clip_model.config.output_attentions
        output_hidden_states = clip_model.config.output_hidden_states
        return_dict = clip_model.config.use_return_dict

        with torch.autocast(device_type="cuda"):
            # get text_outputs
            text_outputs = clip_model.text_model(
                input_ids=input_ids,
                attention_mask=inp["attention_mask"],
                position_ids=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = text_outputs.last_hidden_state
            text_embeds = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=inp["input_ids"].device
                ),
                (inp["input_ids"] == eos_token_id).int().argmax(dim=-1),
            ]
            text_embeds = clip_model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            return text_embeds


def alt_img_forward(clip_model, clip_processor, imgs, v2=True):
    # this is a modified version of inference_utils copy with autograd on for CLIP
    # Use the line below to preprocess PIL images:
    # imgs = torch.stack([clip_processor(img) for img in imgs]).cuda()
    imgs = imgs.cuda()
    if v2:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            image_embeds = clip_model.encode_image(imgs)
            image_embeds_norm = image_embeds.norm(dim=-1, keepdim=True)
            image_embeds = image_embeds / image_embeds_norm
            return image_embeds
    else:
        output_attentions = clip_model.config.output_attentions
        output_hidden_states = clip_model.config.output_hidden_states
        return_dict = clip_model.config.use_return_dict

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            vision_outputs = clip_model.vision_model(
                pixel_values=imgs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            image_embeds = vision_outputs[1]
            image_embeds = clip_model.visual_projection(image_embeds)

            # normalized features
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            return image_embeds


def run_textual_inversion_plus(
    train_path,
    tokens,
    z_objects,
    betas,
    confidences,
    z_names,
    weights,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2",
    max_train_steps=3000,
    learning_rate=5.0e-04,
):
    """
    train_path = path to dataset with all confounded images
    tokens = list of custom tokens for z*
    z_objects = list of indices of objects (classes) z [used for file subpath]
    betas = list of strings for the background in the prompt [used for file subpath]
        each image subpath looks like (train_path\beta\z_object\00.png)
    z_names = list of class names (plain text) corresponding to objects
    """
    betas_unique = list(set(betas))
    # Fixed Parameters
    learnable_property = "object"
    scale_lr = False
    lr_scheduler = "constant"
    lr_warmup_steps = 0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    repeats = 1
    # revision = "fp16"
    revision = "main"
    mixed_precision = "fp16"
    tokenizer_name = None
    resolution = 768
    center_crop = True
    train_batch_size = 1
    gradient_accumulation_steps = 1
    local_rank = -1
    pin_mags = 1
    save_intermediates = True
    logging_dir = "logs"

    confidences = torch.Tensor(confidences).unsqueeze(-1)

    def get_z_beta_path(z_idx, beta):
        path_string = os.path.join(train_path, "_".join(beta.split()), str(z_idx))
        if os.path.exists(path_string):
            return path_string
        else:
            raise RuntimeError(
                f"images for the (z, beta) pair ({z_idx}, {beta}) do not exist at the expected location {path_string} "
            )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # mixed_precision=mixed_precision,
        # log_with="tensorboard",
        # logging_dir=logging_dir,
    )

    # Load the tfokenizer and add the placeholder token as a additional special token
    if tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    elif pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    # text_encoder = text_encoder.to(torch.float16)

    initializer_token_standins = [token + "_init" for token in tokens]
    # Convert the initializer_token, placeholder_token to ids
    initializer_token_ids = [
        utils.load_initializer_text(
            text_encoder, tokenizer, class_name, initializer_token_standin
        )
        for class_name, initializer_token_standin in zip(
            z_names, initializer_token_standins
        )
    ]
    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(tokens)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the tokens {tokens}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Load models and create wrapper for stable diffusion
    curr_emb = text_encoder.text_model.embeddings.token_embedding
    num_orig_embs = curr_emb.num_embeddings
    emb_dim = curr_emb.embedding_dim

    assert placeholder_token_ids[-1] == num_added_tokens + num_orig_embs - 1
    # we added more than one (in contrast to original Textual Inversion code)

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for placeholder_token_id, initializer_token_id in zip(
        placeholder_token_ids, initializer_token_ids
    ):
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    """
    In the following block we are trying to create a new embedding space split into two parts:
    - the pre-existing learnt part
    - the auxillary part which needs to be learnt

    """
    if pin_mags == 1:
        mag_targets = []
        for initializer_token_id in initializer_token_ids:
            mag_targets.append(torch.tensor([initializer_token_id]))
        mag_targets = torch.cat(mag_targets, dim=0)
    else:
        mag_targets = None
    new_emb = utils.SplitEmbedding(
        num_orig_embs, num_added_tokens, emb_dim, magnitude_targets=mag_targets
    )
    new_emb.initialize_from_embedding(
        text_encoder.text_model.embeddings.token_embedding
    )
    text_encoder.text_model.embeddings.token_embedding = new_emb
    """
    now the text_encoder token embeddings above are a SplitEmbedding object
    in this, aux_embedding is the set of learnable embeddings
    """

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
        text_encoder.text_model.embeddings.token_embedding.main_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )
    """
    even though we are passing all embeddings to the optimizer,
    the main embedding portion of the split embedding has been frozen above
    """
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    data_for_z_beta_pairs = {}
    for z_idx, beta, token in zip(z_objects, betas, tokens):
        z_beta_path = get_z_beta_path(z_idx, beta)
        # to handle repeated z
        # assumes each (z,beta) pair occurs atmost once
        if z_idx not in data_for_z_beta_pairs.keys():
            data_for_z_beta_pairs[z_idx] = {beta: {"path": z_beta_path}, "token": token}
        else:
            data_for_z_beta_pairs[z_idx][beta] = {"path": z_beta_path}

    for z_idx, beta in zip(z_objects, betas):
        z_beta_path = data_for_z_beta_pairs[z_idx][beta]["path"]
        token = data_for_z_beta_pairs[z_idx]["token"]
        train_dataset = TextualInversionPlusDataset(
            data_root=z_beta_path,
            tokenizer=tokenizer,
            size=resolution,
            placeholder_token=token,
            repeats=repeats,
            learnable_property=learnable_property,
            center_crop=center_crop,
            set="train",
            background_text=beta,
            all_background_texts=betas_unique,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        data_for_z_beta_pairs[z_idx][beta]["dataset"] = train_dataset
        data_for_z_beta_pairs[z_idx][beta]["dataloader"] = train_dataloader

    # create rho^ = G_phi(z, beta) network
    class GPhi(nn.Module):
        def __init__(self, embedding_size, num_classes, tokenizer, text_encoder):
            """
            concatenates z and beta embeddings
            passes them through a linear model to predict
            the vector of confidence values
            """
            super(GPhi, self).__init__()
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
            self.linear = nn.Linear(2 * embedding_size, 1)

        def forward(self, z_token, beta_token):
            z_emb = self.embedding_from_token(z_token)
            beta_emb = self.embedding_from_token(beta_token)
            concat_emb = torch.cat((z_emb, beta_emb), dim=1)
            output = self.linear(concat_emb)
            return output

        def embedding_from_token(self, token):
            text_encoder = self.text_encoder
            tokenizer = self.tokenizer
            return text_encoder.get_input_embeddings().weight.data[
                tokenizer.convert_tokens_to_ids(token)
            ]

    gphi = GPhi(
        embedding_size=1024,
        num_classes=len(z_names),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    gphi.to(accelerator.device)
    confidences = confidences.to(accelerator.device)
    gphi_optimizer = torch.optim.AdamW(
        gphi.parameters(),  # besides auxiliary embeddings and linear layer other params were frozen earlier
        lr=learning_rate,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    (
        text_encoder,
        optimizer,
        train_dataloader,
        lr_scheduler,
        gphi_optimizer,
    ) = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler, gphi_optimizer
    )

    # Move vae and unet to device
    # vae.to(accelerator.device, torch.float16)
    # unet.to(accelerator.device, torch.float16)

    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion")
    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    # return Stable Diffusion Model
    pipe = infer_utils.get_pipe_fp32(text_encoder=text_encoder, tokenizer=tokenizer)
    w = np.array(weights)
    w = w / np.sum(w)
    wandb.init(project="Textual Inversion Plus")
    for epoch in range(num_train_epochs):
        text_encoder.train()
        losses = {}
        for z_idx, beta in zip(z_objects, betas):
            train_dataloader = data_for_z_beta_pairs[z_idx][beta]["dataloader"]
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    if revision == "fp16":
                        batch["pixel_values"] = (
                            batch["pixel_values"]
                            .type(torch.float16)
                            .to(accelerator.device)
                        )
                    else:
                        batch["pixel_values"] = batch["pixel_values"].to(
                            accelerator.device
                        )

                    # Convert images to latent space
                    latents = (
                        vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                    )
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device, torch.float16)

                    if revision == "fp16":
                        noise = noise.type(torch.float16)

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning (only for the prompt with the true beta (same as real image))
                    encoder_hidden_states = text_encoder(
                        batch["input_ids_list"][batch["true_beta_index"]]
                    )[0]

                    if revision == "fp16":
                        encoder_hidden_states = encoder_hidden_states.type(
                            torch.float16
                        )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    loss_1 = (
                        F.mse_loss(model_pred, target, reduction="none")
                        .mean([1, 2, 3])
                        .mean()
                    )

                    loss_1 = w[0] * loss_1

                    batch["text_prompts"] = [
                        prompt[0] for prompt in batch["text_prompts"]
                    ]
                    if epoch % 10 == 0:
                        loss_2 = clip_contrastive_loss(
                            prompts=batch["text_prompts"],
                            pipe=pipe,
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            random_seed=epoch,
                        )  # contrastive term
                        loss_2_cpu = loss_2.cpu()
                    else:
                        loss_2 = 0
                        loss_2_cpu = loss_2

                    loss_2 = w[1] * loss_2

                    loss = loss_1 + loss_2

                    losses[z_idx] = {
                        "loss": loss.cpu(),
                        "loss_1": loss_1.cpu(),
                        "loss_2": loss_2_cpu,
                    }

                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if global_step >= max_train_steps:
                    break

        # optimize g-phi
        beta_one_word_tokens = [beta.split()[-1] for beta in betas]
        predictions = gphi(z_token=tokens, beta_token=betas)
        loss_3 = F.mse_loss(predictions, confidences)
        loss_3_scaled = w[2] * loss_3
        accelerator.backward(loss_3_scaled)
        gphi_optimizer.step()
        gphi_optimizer.zero_grad()

        losses["loss_3"] = loss_3_scaled.cpu()

        accelerator.wait_for_everyone()
        print(losses)
        log_dict = {
            f"loss_{z_idx}": {
                "loss": losses[z_idx]["loss"],
                "loss_1": losses[z_idx]["loss_1"],
                "loss_2": losses[z_idx]["loss_2"],
            }
            for z_idx in z_objects
        }

        log_dict["loss_3"] = losses["loss_3"]
        wandb.log(log_dict)

    emb_weight = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight

    learned_embeds = emb_weight[placeholder_token_ids].detach().cpu()

    beta_one_word_tokens = [beta.split()[-1] for beta in betas]
    predictions = gphi(z_token=tokens, beta_token=betas)
    cnf_error = F.mse_loss(predictions, confidences)

    accelerator.end_training()
    wandb.finish()
    return learned_embeds, cnf_error
