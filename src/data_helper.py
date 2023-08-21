from abc import ABC, abstractmethod, abstractproperty
from torch.utils.data import DataLoader, Dataset, Subset
import constants as constants
import numpy as np
from src import dataset as ds
from utils import common_utils as cu


def collate_fn(batch):
    return tuple(zip(*batch))


class DataHelper(ABC):
    def __init__(
        self,
        dataset_name,
        dataset_type,
        trn_ds=None,
        val_ds=None,
        tst_ds=None,
        trntst_ds=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

        self.trn_ds: ds.AbsDataset = trn_ds
        self.val_ds: ds.AbsDataset = val_ds
        self.tst_ds: ds.AbsDataset = tst_ds
        self.trntst_ds: ds.AbsDataset = trntst_ds

        self.batch_size = kwargs.get(constants.BATCH_SIZE, 8)
        self.num_workers = kwargs.get(constants.NUM_WORKERS, 4)

        self.__set_loaders()

        pass

    def __str__(self) -> str:
        name = self._name
        if self.trn_ds is not None:
            name += f" Trn: {len(self.trn_ds)}"
        if self.val_ds is not None:
            name += f" Val: {len(self.val_ds)}"
        if self.tst_ds is not None:
            name += f" Tst: {len(self.tst_ds)}"
        return name

    @property
    def _dataset_name(self):
        return self.dataset_name

    @property
    def _dataset_type(self):
        return self.dataset_type

    @property
    def _name(self):
        return f"{self.dataset_name}"

    @property
    def _trn_loader(self) -> DataLoader:
        return self.trn_loader

    @property
    def _val_loader(self) -> DataLoader:
        return self.val_loader

    @property
    def _tst_loader(self) -> DataLoader:
        return self.tst_loader

    @property
    def _trntst_loader(self) -> DataLoader:
        assert (
            self.trntst_ds is not None
        ), "Please pass the trntst_ds to the Datahelper even though it may be redundant"
        return self.trn_tst_loader

    @property
    def _trn_ds(self) -> ds.AbsDataset:
        return self.trn_ds

    @property
    def _sim_ds(self) -> ds.AbsDataset:
        return self.sim_ds

    @property
    def _trntst_ds(self) -> ds.AbsDataset:
        return self.trntst_ds

    @property
    def _val_ds(self) -> ds.AbsDataset:
        return self.val_ds

    @property
    def _tst_ds(self) -> ds.AbsDataset:
        return self.tst_ds

    def __set_loaders(self):
        """Assigns the Data loaders for the dataset"""
        if self.trn_ds is not None:
            self.trn_loader = DataLoader(
                self.trn_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.trn_tst_loader = DataLoader(
                self.trntst_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        if self.val_ds is not None:
            self.val_loader = DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        if self.tst_ds is not None:
            self.tst_loader = DataLoader(
                self.tst_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        constants.logger.info(f"{self.__str__()}")

    @property
    def _num_classes(self) -> int:
        if self.trn_ds is not None:
            return self.trn_ds._num_classes
        else:
            raise NotImplementedError("Base class must override me")

    @property
    def _classes(self) -> np.ndarray:
        if self.trn_ds is not None:
            return self.trn_ds._classes
        else:
            raise NotImplementedError("Base class must override me")

    @property
    def _trn_image_files(self) -> np.ndarray:
        return self.trn_ds._image_files

    @property
    def _val_image_files(self) -> np.ndarray:
        return self.val_ds._image_files

    @property
    def _tst_image_files(self) -> np.ndarray:
        return self.tst_ds._image_files

    @property
    def _trn_labels(self) -> np.ndarray:
        return self.trn_ds._image_labels

    @property
    def _val_labels(self) -> np.ndarray:
        return self.val_ds._image_labels

    @property
    def _tst_labels(self) -> np.ndarray:
        return self.tst_ds._image_labels

    def assign_val_ds(self, val_size):
        """Partitions the train dataset into train and val

        Args:
            val_size (_type_): _description_
        """
        cu.set_seed(constants.seed)
        indices = np.random.permutation(len(self.trn_ds)).tolist()
        trn_idxs, val_idxs = sorted(indices[:-val_size]), sorted(indices[-val_size:])

        # copy the trn dataset to a new dataset
        trn_ds_all = self.trn_ds
        self.trn_ds = Subset(trn_ds_all, trn_idxs)
        self.val_ds = Subset(trn_ds_all, val_idxs)

        self.__set_loaders()


class RecDataHelper(DataHelper):
    def __init__(
        self,
        dataset_name,
        dataset_type,
        trn_ds: Dataset = None,
        val_ds: Dataset = None,
        tst_ds: Dataset = None,
        trntst_ds: Dataset = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_name,
            dataset_type,
            trn_ds=trn_ds,
            val_ds=val_ds,
            tst_ds=tst_ds,
            trntst_ds=trntst_ds,
            **kwargs,
        )
