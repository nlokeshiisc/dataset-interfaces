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
    def _unique_betas(self) -> np.ndarray:
        return np.unique(self._image_betas, axis=0)

    @property
    def _num_betas(self) -> int:
        return len(self._unique_betas)

    @property
    def _image_betas(self) -> np.ndarray:
        return np.asarray(
            [self.get_beta(idx=idx) for idx in range(len(self))], dtype=np.int32
        )

    @property
    def _image_objnames(self) -> np.ndarray:
        objnames = [self.get_objname(idx=idx) for idx in range(len(self))]
        return np.asarray(objnames, dtype=str)

    @property
    def _unq_image_objnames(self) -> np.ndarray:
        return np.unique(self._image_objnames)

    @property
    def _clswise_image_objnames(self) -> dict:
        clswise_image_objnames = {}
        for cls in self._classes:
            clswise_image_objnames[cls] = np.unique(
                self._image_objnames[
                    self.get_cls_idxs(labels=self._image_labels, cls=cls)
                ]
            )
        return clswise_image_objnames

    def get_beta(self, idx=None, image_file=None) -> np.ndarray:
        assert (idx is not None) or (
            image_file is not None
        ), "Either idx or image_file must be provided"
        if idx is not None:
            image_file = self._image_files[idx]
        beta = image_file[-7:-4]
        return np.asarray([eval(beta[0]), eval(beta[1]), eval(beta[2])], dtype=np.int32)

    def get_cls_idxs(self, labels, cls) -> np.ndarray:
        return np.where(labels == cls)[0]

    def get_objname(self, idx=None, image_file=None) -> str:
        assert (idx is not None) or (
            image_file is not None
        ), "Either idx or image_file must be provided"
        if idx is not None:
            image_file = self._image_files[idx]
        image_file = Path(image_file)
        return image_file.name[:-8]

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
        self.betas = self._image_betas
        self.objnames = self._image_objnames

    @property
    def _betas(self) -> np.ndarray:
        return self.betas

    @property
    def _objnames(self) -> np.ndarray:
        return self.objnames

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        image_label = self._image_labels[idx]
        image = dsu.load_color_image(image_file)
        image = image / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label


class ClsDataset(AbsDataset, ABC):
    def __init__(
        self,
        dataset_name,
        dataset_split,
        df: pd.DataFrame = None,
        collapse_labels=False,
        collapse_dict=None,
        transform=None,
        **kwargs,
    ):
        super().__init__(dataset_name, dataset_split, df, transform, **kwargs)
        self.collapse_dict = collapse_dict
        self.collapsed = collapse_labels
        self.fine_labels = None
        self.num_fine_classes = None
        self.process_collapsed_dict(**kwargs)
        if dataset_split in [constants.TRN, constants.TRNTST]:
            self.load_cls_trn = kwargs.get(constants.LOAD_CLS_TRN, False)

    def process_collapsed_dict(self, **kwargs):
        if self.collapsed == True:
            assert self.collapse_dict is not None, "collapse_dict is None"
            assert len(self.collapse_dict.keys()) == len(
                np.unique(self._df["labels"].values)
            ), f"Expecxted collapsed dict to have mapping for {np.unique(self._df['labels'].values)} but got only for {self.collapse_dict.keys()}"
            self.fine_labels = self.df["labels"].values  # Fine grain labels
            self.num_fine_classes = len(np.unique(self.fine_labels))
            self.labels = np.asarray([self.collapse_dict[l] for l in self.fine_labels])
            self.classes = np.unique(self.labels)

    @property
    def _collapsed(self) -> bool:
        return self.collapsed

    @property
    def _collapsed_dict(self) -> dict:
        return self.collapse_dict

    @property
    def _collapsed_dict_str(self) -> str:
        return constants.dict_to_str(self.collapse_dict)

    @property
    def _fine_labels(self) -> np.ndarray:
        """Returns the fine grain labels accodciated with the dataset before collapsing

        Returns:
            np.ndarray: _description_
        """
        if self.fine_labels is None:
            return self._image_labels
        return self.fine_labels

    @property
    def _num_fine_classes(self) -> int:
        return self.num_fine_classes

    @abstractproperty
    def _all_betas(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()


class ShapenetClsDataset(ClsDataset):
    def __init__(
        self,
        dataset_name,
        dataset_split,
        df: pd.DataFrame = None,
        collapse_labels=False,
        collapse_dict=None,
        transform=None,
        **kwargs,
    ):
        super().__init__(
            dataset_name,
            dataset_split,
            df,
            collapse_labels,
            collapse_dict,
            transform,
            **kwargs,
        )

    @property
    def _all_betas(self) -> np.array:
        return constants.SHAPENET_BETAS

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        label = self._image_labels[idx]
        image = dsu.load_color_image(image_file)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.Tensor([label]).long()
        beta = (
            torch.Tensor(self.get_beta(idx=idx)).view(1, -1).long()
        )  # make beta a row vector
        return idx, image, beta, label


class MedicalClsDataset(ShapenetClsDataset):
    """For now i do not anticipate any new changes for medical dataset.
    So simply overriding the ShapenetClsDataset should suffice.
    """

    def __init__(
        self,
        dataset_name,
        dataset_split,
        df: pd.DataFrame = None,
        collapse_labels=False,
        collapse_dict=None,
        transform=None,
        **kwargs,
    ):
        super().__init__(
            dataset_name,
            dataset_split,
            df,
            collapse_labels,
            collapse_dict,
            transform,
            **kwargs,
        )

    @property
    def _all_betas(self) -> np.array:
        return constants.MEDICAL_BETAS

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class VerDataset(AbsDataset, ABC):
    def __init__(self, cls_ds: ClsDataset, rho, **kwargs):
        super().__init__(
            cls_ds._dataset_name,
            cls_ds._dataset_split,
            cls_ds._df,
            cls_ds.transform,
            **kwargs,
        )
        self.cls_ds = cls_ds
        assert len(rho) == len(cls_ds._image_files)
        self.rho = rho
        self.grp_rho = None
        self.fine_grp_rho = None
        self.use_imgemb = kwargs.get(constants.USE_IMGEMB, False)
        self.img_emb = None

    @property
    def _image_files(self):
        return self.cls_ds._image_files

    @property
    def _image_labels(self):
        return self.cls_ds._image_labels

    @property
    def _image_betas(self):
        return self.cls_ds._image_betas

    @property
    def _image_fine_labels(self):
        return self.cls_ds._fine_labels

    @abstractproperty
    def _z_vals(self):
        """Pass a list of z dims"""
        raise NotImplementedError()

    @abstractproperty
    def _beta_vals(self):
        """Pass a list of beta values"""
        raise NotImplementedError()

    @abstractproperty
    def _num_betas(self):
        """Returns the number of distinct betas in the dataset"""
        raise NotImplementedError()

    @abstractproperty
    def _all_betas(self):
        raise NotImplementedError()

    @abstractproperty
    def _z_str(self):
        """pass a list mentioning what each z dim corresponds to"""
        raise NotImplementedError()

    @abstractproperty
    def _beta_str(self):
        """pass a list mentioning what each beta dim corresponds to"""
        raise NotImplementedError()

    @property
    def _rho(self) -> np.ndarray:
        return self.rho

    @property
    def _grp_rho(self):
        return self.grp_rho

    @_grp_rho.setter
    def _grp_rho(self, grp_rho):
        self.grp_rho = grp_rho

    @property
    def _fine_grp_rho(self):
        return self.fine_grp_rho

    @_fine_grp_rho.setter
    def _fine_grp_rho(self, fine_grp_rho):
        self.fine_grp_rho = fine_grp_rho

    @property
    def _img_emb(self):
        return self.img_emb

    @_img_emb.setter
    def _img_emb(self, img_emb):
        assert len(img_emb) == len(self._image_files)
        self.img_emb = img_emb

    @abstractmethod
    def y_beta_rho(self, y, beta: list) -> float:
        raise NotImplementedError()

    @abstractmethod
    def z_beta_rho(self, z, beta: list) -> float:
        raise NotImplementedError()

    @abstractmethod
    def sample_z(self):
        """Sample an arbitrary z value"""
        raise NotImplementedError()

    @abstractmethod
    def sample_beta(self) -> torch.LongTensor:
        """Sample an arbitrary beta value"""
        raise NotImplementedError()

    def __len__(self):
        return len(self._image_files)


class ShapenetVerDataset(VerDataset):
    def __init__(self, cls_ds: ClsDataset, rho, **kwargs):
        super().__init__(cls_ds, rho, **kwargs)

    def __getitem__(self, idx):
        z, beta = self._image_fine_labels[idx], self._image_betas[idx]
        y = self._image_labels[idx]
        imgemb = torch.empty(0)
        if self.use_imgemb == True:
            imgemb = self._img_emb[idx]
        rho = self._rho[idx]
        fine_grp_rho = self.z_beta_rho(z=z, beta=beta)
        grp_rho = self.y_beta_rho(y=y, beta=beta)
        return torch.LongTensor([idx]), {
            "z": torch.LongTensor([z]),
            constants.BETA: torch.LongTensor(beta),
            "y": torch.LongTensor([y]),
            constants.IMGEMB: imgemb,
            constants.RHO: torch.Tensor([rho]),
            constants.GRP_RHO: torch.Tensor([grp_rho]),
            constants.FINE_GRP_RHO: torch.Tensor([fine_grp_rho]),
        }

    @property
    def _z_vals(self) -> list:
        """Pass a list of z dims"""
        return [10]

    @property
    def _beta_vals(self):
        """Pass a list of beta values"""
        return [6, 3, 4]

    @property
    def _z_str(self):
        """pass a list mentioning what each z dim corresponds to"""
        return ["fine_labels"]

    @property
    def _beta_str(self):
        """pass a list mentioning what each beta dim corresponds to"""
        return [constants.VIEW, constants.ZOOM, constants.LGT_COLOR]

    @property
    def _num_betas(self):
        return constants.SHAPENET_BETAS.shape[0]

    @property
    def _all_betas(self):
        return constants.SHAPENET_BETAS

    @property
    def _beta_to_idx_dict(self):
        return constants.shapenet_beta_to_idx_dict

    def sample_z(self):
        return torch.LongTensor([np.random.choice(10)])

    def sample_beta(self):
        rnd_idx = np.random.choice(len(constants.SHAPENET_BETAS))
        return torch.LongTensor([constants.SHAPENET_BETAS[rnd_idx]])

    def y_beta_rho(self, y, beta: list) -> float:
        if type(beta) is not str:
            beta = constants.betatostr_fn(beta)
        return self.grp_rho[y][self._beta_to_idx_dict[beta]]

    def z_beta_rho(self, z, beta: list) -> float:
        if type(beta) is not str:
            beta = constants.betatostr_fn(beta)
        return self.fine_grp_rho[z][self._beta_to_idx_dict[beta]]


class MedicalVerDataset(ShapenetVerDataset):
    def __init__(self, cls_ds: ClsDataset, rho, **kwargs):
        super().__init__(cls_ds, rho, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    @property
    def _z_vals(self) -> list:
        """Pass a list of z dims"""
        return [7]

    @property
    def _beta_vals(self):
        """Pass a list of beta values"""
        return [3, 3, 3]

    @property
    def _z_str(self):
        """pass a list mentioning what each z dim corresponds to"""
        return ["fine_labels"]

    @property
    def _beta_str(self):
        """pass a list mentioning what each beta dim corresponds to"""
        return [constants.ZOOM, constants.ILLUMINATION, constants.CONTRAST]

    @property
    def _num_betas(self):
        return constants.MEDICAL_BETAS.shape[0]

    @property
    def _all_betas(self):
        return constants.MEDICAL_BETAS

    def sample_z(self):
        return torch.LongTensor([np.random.choice(7)])

    def sample_beta(self):
        rnd_idx = np.random.choice(self._num_betas)
        return torch.LongTensor([constants.MEDICAL_BETAS[rnd_idx]])

    @property
    def _beta_to_idx_dict(self):
        return constants.medical_beta_to_idx_dict


class ShapenetRecDataset(ShapenetVerDataset):
    """The recourse dataset and the verifier dataset more or less are similar.
    Thus for now, it would suffice just to overrisde the ShapenetVerDataset.
    TODO: Later, see if we need to create a dedicated abstract class for the RecDataset.

    Args:
        ShapenetVerDataset (_type_): _description_
    """

    def __init__(self, cls_ds: ClsDataset, rho, **kwargs):
        super().__init__(cls_ds, rho, **kwargs)
        self.use_imgemb = True  # For recourse, we always use image embedding

    @property
    def _beta_to_idx_dict(self):
        return constants.shapenet_beta_to_idx_dict

    def __getitem__(self, idx):
        z, beta = self._image_fine_labels[idx], self._image_betas[idx]
        y = self._image_labels[idx]
        imgemb = self._img_emb[idx]
        rho = self._rho[idx]
        beta_id = self._beta_to_idx_dict[constants.betatostr_fn(beta)]

        return torch.LongTensor([idx]), {
            "z": torch.LongTensor([z]),
            constants.BETA: torch.LongTensor(beta),
            constants.BETAID: torch.LongTensor([beta_id]),
            "y": torch.LongTensor([y]),
            constants.IMGEMB: imgemb,
            constants.RHO: torch.Tensor([rho]),
        }


class MedicalRecDataset(MedicalVerDataset):
    def __init__(self, cls_ds: ClsDataset, rho, **kwargs):
        super().__init__(cls_ds, rho, **kwargs)
        self.use_imgemb = True  # For recourse, we always use image embedding

    @property
    def _beta_to_idx_dict(self):
        return constants.medical_beta_to_idx_dict

    def __getitem__(self, idx):
        z, beta = self._image_fine_labels[idx], self._image_betas[idx]
        y = self._image_labels[idx]
        imgemb = self._img_emb[idx]
        rho = self._rho[idx]
        beta_id = self._beta_to_idx_dict[constants.betatostr_fn(beta)]

        return torch.LongTensor([idx]), {
            "z": torch.LongTensor([z]),
            constants.BETA: torch.LongTensor(beta),
            constants.BETAID: torch.LongTensor([beta_id]),
            "y": torch.LongTensor([y]),
            constants.IMGEMB: imgemb,
            constants.RHO: torch.Tensor([rho]),
        }
