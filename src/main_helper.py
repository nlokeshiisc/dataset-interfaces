import torch
from torchvision.models import resnet50
from dataset_interfaces import utils, imagenet_utils
import constants as constants
from torch.utils.data import DataLoader
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import pandas as pd


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


class AccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.total = 0

        self.cls_acc = defaultdict(int)
        self.cls_total = defaultdict(int)
        self.confusion_preds = defaultdict(int)

    def reset(self):
        self.correct = 0
        self.total = 0
        self.cls_acc = defaultdict(int)
        self.cls_total = defaultdict(int)
        self.confusion_preds = defaultdict(lambda: defaultdict(int))

    def update(self, y_preds, y):
        pred_labels = y_preds.argmax(dim=1)
        correct = pred_labels == y
        self.correct += correct.sum().item()
        self.total += len(y)

        # update classwise accuracy
        for i in range(len(y)):
            cls = y[i].item()
            self.cls_acc[cls] += correct[i].item()
            self.cls_total[cls] += 1

        # update confusion matrix
        for i in range(len(y)):
            pred = pred_labels[i].item()
            cls = y[i].item()
            self.confusion_preds[cls][pred] += 1

    def accuracy(self, verbose=False):
        acc = self.correct / self.total
        if verbose:
            print(f"Accuracy: {acc}")
        return acc

    def classwise_accuracy(self, verbose=False):
        cls_acc = {}
        for cls in self.cls_acc:
            cls_acc[cls] = self.cls_acc[cls] / self.cls_total[cls]
        if verbose:
            pprint(cls_acc)
        return cls_acc

    def confusion_matrix(self, verbose=False):
        confusion = {}
        for y in self.confusion_preds:
            confusion[y] = {}
            total = self.cls_total[y]

            for pred in self.confusion_preds[y]:
                confusion[y][pred] = self.confusion_preds[y][pred] / total
        if verbose:
            pprint(confusion)
        return confusion


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: str, cache=False
):
    acc_meter = AccuracyMeter()
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
    acc_meter = AccuracyMeter()
    acc_meter.reset()
    true_labels = df["true"].values
    pred_labels = df["pred"].values
    acc_meter.update(y_preds=torch.Tensor(pred_labels), y=torch.Tensor(true_labels))
    print(f"Accuracy: {acc_meter.accuracy()}")
    print(f"Classwise accuracy: {acc_meter.classwise_accuracy()}")
    return acc_meter
