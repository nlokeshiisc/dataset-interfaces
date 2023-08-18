import torch
from torchvision.models import resnet50
from dataset_interfaces import utils, imagenet_utils
import constants as constants
from torch.utils.data import DataLoader
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from utils import torch_utils as tu


def get_model(model_name, pretrn=True):
    if model_name == "resnet50":
        model = resnet50(pretrained=pretrn)
    return model


def get_ds_dl(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    ds = utils.ImageNet_Star_Dataset(
        path=constants.imagenet_star_dir,
        shift=dataset_name,
        # mask_path=constants.imagenet_star_dir / "masks.npy",
        transform=constants.RESNET_TRANSFORMS[constants.TST],
    )
    # Create dataloader
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    return ds, dl


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: str, cache=False
):
    acc_meter = tu.AccuracyMeter()
    acc_meter.reset()
    model = model.to(device)
    model.eval()
    pbar = tqdm(loader, total=len(loader))
    true_labels, pred_labels, cnf = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_preds = model(x)

            true_labels.append(y.cpu())
            pred_labels.append(y_preds.argmax(dim=1).cpu())
            cnf.append(y_preds.gather(1, y.view(-1, 1)).squeeze().cpu())

            acc_meter.update(y_preds, y)
            pbar.set_postfix({"Accuracy": acc_meter.accuracy()})
            pbar.update(1)

        if cache == True:
            shift = loader.dataset.shift
            true_labels = torch.cat(true_labels).numpy()
            pred_labels = torch.cat(pred_labels).numpy()
            cnf = torch.cat(cnf).numpy()
            df = pd.DataFrame({"true": true_labels, "pred": pred_labels, "cnf": cnf})
            df.to_csv(constants.PROJ_DIR / "cache" / f"{shift}_preds.csv", index=False)

        return acc_meter


def df_to_acc(dataset_name):
    try:
        df = pd.read_csv(constants.PROJ_DIR / "cache" / f"{dataset_name}_preds.csv")
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
