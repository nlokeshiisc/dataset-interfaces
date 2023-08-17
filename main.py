from dataset_interfaces import utils, imagenet_utils
import torch
import constants as constants
from pathlib import Path
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict


resnet = resnet50(pretrained=True)
resnet.eval()

# Get resnet50 transform
transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are the mean and std from imagenet
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
base = utils.ImageNet_Star_Dataset(
    path=constants.imagenet_star_dir,
    shift=constants.BASE,
    mask_path=constants.imagenet_star_dir / "masks.npy",
    transform=None,
)
print(f"Base dataset has {len(base)} images")

# Create dataloader
dataloader = DataLoader(base, batch_size=32, shuffle=False, num_workers=0)


class AccuracyMeter:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.cls_acc = defaultdict(int)
        self.cls_total = defaultdict(int)

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, y_preds, y):
        correct = y_preds.argmax(dim=1) == y
        self.correct += correct.sum().item()
        self.total += len(y)

        # Also update classwise accuracy
        for i in range(len(y)):
            cls = y[i].item()
            self.cls_acc[cls] += correct[i].item()
            self.cls_total[cls] += 1

    def compute(self):
        return self.correct / self.total

    def compute_cls(self):
        cls_acc = {}
        for cls, total in self.cls_total.items():
            cls_acc[cls] = self.cls_acc[cls] / total
        return cls_acc


# Evaluate base on dataloader
resnet.cuda()
resnet.eval()
acc_meter = AccuracyMeter()
acc_meter.reset()
with torch.no_grad():
    for i, (x, y) in enumerate(dataloader):
        x, y = x.cuda(), y.cuda()
        y_preds = resnet(x)
        correct = (y_preds.argmax(dim=1) == y).sum().item()
        acc_meter.update(correct, len(y))
    print(f"Accuracy: {acc_meter.compute()}")
    print(f"Classwise accuracy: {acc_meter.compute_cls()}")

"""
print(base[0][0].shape)

night = utils.ImageNet_Star_Dataset(
    path=constants.imagenet_star_dir,
    shift=constants.NIGHT,
    mask_path=constants.imagenet_star_dir / "masks.npy",
    transform=None,
)
"""
