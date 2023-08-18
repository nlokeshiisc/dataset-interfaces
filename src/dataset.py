from torch.utils.data import Dataset
from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
from utils import dataset_utils as dsu
import torch
import constants
from pathlib import Path


class AbsDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_name,
        dataset_split,
        df: pd.DataFrame = None,
        transform=None,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.df = df
        self.image_files = self.df["image_files"].values
        self.labels = self.df["labels"].values
        self.src_shifts = self.df["shifts"].values

        self.transform = transform
        self.classes = np.unique(self.labels)
        pass

    @property
    def _dataset_name(self):
        return self.dataset_name

    @property
    def _dataset_split(self) -> str:
        return self.dataset_split

    @property
    def _name(self):
        return self.dataset_name

    @property
    def _df(self) -> pd.DataFrame:
        assert self.df is not None, "df is None"
        return self.df

    @property
    def _image_files(self) -> np.ndarray:
        return self.image_files

    @property
    def _image_labels(self) -> np.ndarray:
        return self.labels

    @property
    def _classes(self) -> np.ndarray:
        return self.classes

    @property
    def _num_classes(self) -> int:
        return len(self.classes)

    @property
    def _unique_shifts(self) -> np.ndarray:
        return np.unique(self.src_shifts, axis=0)

    @property
    def _num_betas(self) -> int:
        return len(self._unique_shifts)

    def get_cls_idxs(self, labels, cls) -> np.ndarray:
        return np.where(labels == cls)[0]

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        pass


class DfDataset(AbsDataset):
    def __init__(
        self,
        dataset_name,
        dataset_split,
        df: pd.DataFrame = None,
        transform=None,
        **kwargs,
    ):
        super().__init__(dataset_name, dataset_split, df, transform, **kwargs)

    @property
    def _src_shifts(self) -> np.ndarray:
        return self.src_shifts

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        image_label = self._image_labels[idx]
        image_shift = self._src_shifts[idx]
        image = dsu.load_color_image(image_file)
        image = image / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return (
            idx,
            image,
            image_label,
            image_shift,
        )