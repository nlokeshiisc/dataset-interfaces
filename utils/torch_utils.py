from collections import defaultdict
from pprint import pprint
import pandas as pd


class AccuracyMeter:
    def __init__(self, track: list = ["acc", "cls_acc", "confusion"]):
        self.correct = 0
        self.total = 0

        self.cls_acc = None
        self.cls_total = None
        self.confusion_preds = None

        if "cls_acc" in track:
            self.cls_acc = defaultdict(int)
            self.cls_total = defaultdict(int)
        if "confusion" in track:
            self.confusion_preds = defaultdict(int)

    def reset(self):
        self.correct = 0
        self.total = 0
        if self.cls_acc is not None:
            self.cls_acc = defaultdict(int)
            self.cls_total = defaultdict(int)
        if self.confusion_preds is not None:
            self.confusion_preds = defaultdict(lambda: defaultdict(int))

    def update(self, y_preds, y, y_preds_labels=False):
        """_summary_

        Args:
            y_preds (_type_): _description_
            y (_type_): _description_
            pred_labels (bool, optional): Defaults to False. Is y_preds labels or logits?
        """
        if y_preds_labels == False:
            y_preds = y_preds.argmax(dim=1)
        else:
            pass
        correct = y_preds == y
        self.correct += correct.sum().item()
        self.total += len(y)

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
