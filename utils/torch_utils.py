from collections import defaultdict
from pprint import pprint
import torch


class AccuracyMeter:
    def __init__(self, track: list = ["acc", "cls_acc", "confusion"]):
        self.metric = 0
        self.num_samples = 0

        self.cls_acc = None
        self.cls_total = None
        self.confusion_preds = None

        if "cls_acc" in track:
            self.cls_acc = defaultdict(int)
            self.cls_total = defaultdict(int)
        if "confusion" in track:
            self.confusion_preds = defaultdict(int)

    def reset(self):
        self.metric = 0
        self.num_samples = 0
        if self.cls_acc is not None:
            self.cls_acc = defaultdict(int)
            self.cls_total = defaultdict(int)
        if self.confusion_preds is not None:
            self.confusion_preds = defaultdict(lambda: defaultdict(int))

    def update(self, y_preds: torch.Tensor, y: torch.Tensor):
        """_summary_

        Args:
            y_preds (_type_): _description_
            y (_type_): _description_
            pred_labels (bool, optional): Defaults to False. Is y_preds labels or logits?
        """

        assert len(y_preds) == len(y), "y_preds and y must be same length"
        assert y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1), "y must be 1d"

        if y_preds.ndim == 2 and y_preds.shape[1] > 1:
            y_preds = y_preds.argmax(dim=1)

        correct = y_preds == y
        self.metric += correct.sum().item()
        self.num_samples += len(y)

        if self.cls_acc is not None:
            # update classwise accuracy
            for i in range(len(y)):
                cls = y[i].item()
                self.cls_acc[cls] += correct[i].item()
                self.cls_total[cls] += 1

        if self.confusion_preds is not None:
            # update confusion matrix
            for i in range(len(y)):
                pred = y_preds[i].item()
                cls = y[i].item()
                self.confusion_preds[cls][pred] += 1

    def accuracy(self, verbose=False):
        acc = self.metric / self.num_samples
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
