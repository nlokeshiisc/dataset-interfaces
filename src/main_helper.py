import torch
from torchvision.models import resnet50
from dataset_interfaces import utils, imagenet_utils
import constants as constants
from torch.utils.data import DataLoader
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from utils import torch_utils as tu
from utils import common_utils as cu
import itertools
from src import dataset as ds
import numpy as np
from src import data_helper as dh
from pathlib import Path


def get_model(model_name, pretrn=True):
    if model_name == "resnet50":
        model = resnet50(pretrained=pretrn)
    return model


def get_ds_dl(dataset_name, loader=True):
    print(f"Loading dataset: {dataset_name}")
    ds = utils.ImageNet_Star_Dataset(
        path=constants.imagenet_star_dir,
        shift=dataset_name,
        # mask_path=constants.imagenet_star_dir / "masks.npy",
        transform=constants.RESNET_TRANSFORMS[constants.TST],
    )
    if loader == False:
        return ds

    # Create dataloader
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    return ds, dl


@torch.inference_mode()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, cache=False):
    acc_meter = tu.AccuracyMeter()
    acc_meter.reset()
    model = model.to(cu.get_device(), dtype=torch.float16)
    model.eval()

    xent = torch.nn.CrossEntropyLoss(reduction="none")
    sm = torch.nn.Softmax(dim=1)

    pbar = tqdm(loader, total=len(loader))

    true_labels, pred_labels, cnf, image_files, losses = [], [], [], [], []
    with torch.no_grad():
        for idx, path, x, y in loader:
            x, y = x.to(cu.get_device(), dtype=torch.float16), y.to(
                cu.get_device(), dtype=torch.long
            )
            logits = model(x)
            probs = sm(logits)

            loss_perex = xent(logits, y)

            y_preds = logits.argmax(dim=1).to(dtype=torch.long)

            true_labels.append(y.cpu())
            pred_labels.append(y_preds.cpu())
            image_files.append(path)
            losses.append(loss_perex.cpu())
            cnf.append(probs.gather(1, y.view(-1, 1)).squeeze().cpu())

            acc_meter.update(probs, y)
            pbar.set_postfix({"Accuracy": acc_meter.accuracy()})
            pbar.update(1)

        if cache == True:
            shift = loader.dataset.shift
            true_labels = torch.cat(true_labels).numpy()
            pred_labels = torch.cat(pred_labels).numpy()
            image_files = list(itertools.chain(*image_files))
            losses = torch.cat(losses).numpy()
            cnf = torch.cat(cnf).numpy()

            df = pd.DataFrame(
                {
                    constants.IMGFILE: image_files,
                    constants.TRUEY: true_labels,
                    constants.PREDY: pred_labels,
                    constants.CNF: cnf,
                    constants.LOSS: losses,
                }
            )
            df.to_csv(
                constants.PROJ_DIR / "cache" / f"{shift}_preds.csv",
                index=False,
            )

        return acc_meter


def df_to_acc(dataset_name):
    try:
        df = pd.read_csv(
            constants.imagenet_star_dir / "cache" / f"{dataset_name}_preds.csv"
        )
    except:
        print("Call evaluate_model with cache=True first")
        return
    acc_meter = tu.AccuracyMeter()
    acc_meter.reset()
    true_labels = df["true"].values
    pred_labels = df["pred"].values
    acc_meter.update(
        y_preds=torch.Tensor(pred_labels),
        y=torch.Tensor(true_labels),
        y_preds_labels=True,
    )
    return acc_meter


def get_rec_datasets(shifts):
    """Loads the datasets for the shifts and classes specified

    Args:
        shifts (_type_): _description_
        sel_classes (_type_): _description_
    """
    shifts_ds: dict = {}
    for shift in shifts:
        imstar_ds = get_ds_dl(shift, loader=False)
        try:
            shift_df = pd.read_csv(constants.PROJ_DIR / "cache" / f"{shift}_preds.csv")
        except:
            print("Call evaluate_model with cache=True first")
            return
        assert len(shift_df) == len(
            imstar_ds
        ), "Length mismatch between the dataset and the cache dataframe"

        rnd_idx = np.random.choice(len(shift_df))
        img_file = shift_df.iloc[rnd_idx]["image_files"].split("/")[-1]
        ds_img_file = imstar_ds.samples[rnd_idx][0].split("/")[-1]
        assert img_file == ds_img_file, "Image file mismatch"

        cache_ds = ds.DFRho(df=shift_df)

        shifts_ds[shift] = {}
        shifts_ds[shift]["ds"] = imstar_ds
        shifts_ds[shift]["cache"] = cache_ds

    return shifts_ds


def check_df_acc(shifts):
    # This is simply to sanity check the cached dataframes
    shift_ds, acc_meters = {}, {}
    for shift in shifts:
        imstar_ds = get_ds_dl(dataset_name=shift, loader=False)
        shift_ds[shift] = imstar_ds
        acc_meter = df_to_acc(dataset_name=shift)
        acc_meters[shift] = acc_meter

        print(f"Dataset: {shift}, accuracy: {acc_meters[shift].accuracy()}")
    pass


def get_rec_dh(trn_df, tst_df, **kwargs):
    rec_trn_ds = ds.DfDataset(
        dataset_name=constants.IMSTAR,
        dataset_split=constants.TRN,
        df=trn_df,
        transform=constants.RESNET_TRANSFORMS[constants.TRN],
    )
    rec_trntst_ds = ds.DfDataset(
        dataset_name=constants.IMSTAR,
        dataset_split=constants.TRNTST,
        df=trn_df,
        transform=constants.RESNET_TRANSFORMS[constants.TRNTST],
    )
    rec_tst_ds = ds.DfDataset(
        dataset_name=constants.IMSTAR,
        dataset_split=constants.TST,
        df=tst_df,
        transform=constants.RESNET_TRANSFORMS[constants.TST],
    )
    rec_dh = dh.RecDataHelper(
        dataset_name=constants.IMSTAR,
        dataset_type="real",
        trn_ds=rec_trn_ds,
        trntst_ds=rec_trntst_ds,
        tst_ds=rec_tst_ds,
        **kwargs,
    )

    return rec_dh


def check_dir_acc(image_dir: Path):
    """Checks the accuracy of the images in the directory"""
    pass
