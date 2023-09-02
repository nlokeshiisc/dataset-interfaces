import torchvision
import constants as constants
import torch.nn as nn
import torch
from torchvision import models as tv_models
from src import data_helper as dh
from src import data_helper as dh
from src import dataset as ds
import numpy as np
from itertools import product
from pathlib import Path
from utils import common_utils as cu
from abc import ABC, abstractmethod, abstractproperty
import pickle as pkl
from utils import torch_data_utils as tdu


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Embedding(nn.Module):
    def __init__(self, num_vocab, embdim):
        super().__init__()
        self.num_vocab = num_vocab
        self.embdim = embdim
        self.Emb = nn.Embedding(num_vocab, embdim)

    def forward(self, vocab_ids):
        emb = torch.squeeze(self.Emb(vocab_ids))
        if len(vocab_ids) == 1:
            emb = emb.unsqueeze(0)
        return emb

    def emb_size(self):
        return self.embdim

    @property
    def _type(self):
        return "table_emb"

    @property
    def _dir_path(self) -> Path:
        return constants.MODELDIR / "emb" / self._type

    def save_model(self, *, cls_model_name):
        self._dir_path.mkdir(parents=True, exist_ok=True)
        # save the model
        torch.save(
            self.state_dict(),
            self._dir_path / f"{cls_model_name}.pt",
        )
        constants.Logger.info(
            f"Saving emb_model to: {self._dir_path / f'{cls_model_name}.pt'}"
        )

    def load_model(self, cls_model_name):
        self.load_state_dict(
            torch.load(
                self._dir_path / f"{cls_model_name}.pt", map_location=cu.get_device()
            )
        )
        constants.logger.info(
            f"Loaded emb_model from: {self._dir_path / f'{cls_model_name}.pt'}"
        )


class FNN(nn.Module):
    """creates a Feed Forward Neural network with the specified Architecture
    nn ([type]): [description]
    """

    def __init__(self, in_dim, out_dim, nn_arch, prefix, *args, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch
        self.prefix = prefix
        self.batchnorm = kwargs.get(constants.BATCH_NORM, True)

        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch):
            self.model.add_module(
                f"{self.prefix}-beta_hid_{idx}", nn.Linear(prev, hdim)
            )
            self.model.add_module(f"{self.prefix}-ReLU_{idx}", nn.ReLU(inplace=True))
            if self.batchnorm:
                self.model.add_module(f"{self.prefix}-bn_{idx}", nn.BatchNorm1d(hdim))

            prev = hdim
        self.model.add_module(f"{self.prefix}-last_layer", nn.Linear(prev, out_dim))

    def forward(self, input):
        return self.model(input)


# TODO: Correct this class
class RecModel(nn.Module, ABC):
    """This is an abstract class for recourse models.
    For now, we would want to try the TARNET based model.
    """

    def __init__(self, datasets, *args, **kwargs):
        super().__init__()
        self.datasets: dict = datasets
        self.shifts = sorted(self.datasets.keys())
        self.shift_idxs = {shift: idx for idx, shift in enumerate(self.shifts)}
        self.num_shifts = len(self.shifts)

        self.rec_input: list = kwargs.get(constants.INPUT)
        print(f"Input to the recourse model: {self.rec_input}")

        self.embdim = kwargs.get(constants.EMBDIM, 64)
        self.shift_emb = Embedding(num_vocab=len(self.shifts), embdim=self.embdim)
        self.nn_arch = kwargs.get(constants.NN_ARCH, [128, 64])

        if "x" in self.rec_input:
            self.resnet_50 = tv_models.resnet50(pretrained=True)
            self.imgembdim = self.resnet_50.fc.in_features
            self.resnet_50.fc = Identity()
            self.all_dim = self.imgembdim + self.embdim
        if constants.SSTAR in self.rec_input:
            self.sstar_fnn = FNN(
                in_dim=constants.SSTAR_DIM,
                out_dim=self.embdim,
                nn_arch=self.nn_arch,
                prefix="sstar",
            )
            self.all_dim = self.embdim + self.embdim

        self.sm = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    @abstractmethod
    def forward(self, *, img, src_shift, rec_shift):
        raise NotImplementedError()

    @abstractmethod
    def forward_all_beta(
        self,
        *,
        img,
        src_shift,
    ):
        raise NotImplementedError()

    def _rho_cls_model(self, prefix, nn_arch=[128, 64], batch_norm=False) -> nn.Module:
        """Constructs a deep model for rho prediction

        Args:
            prefix : name of the module
            nn_arch (list, optional): . Defaults to [128, 64].
            batch_norm (bool, optional): . Defaults to False.

        Returns:
            nn.Module:
        """
        fnn_args = {constants.BATCH_NORM: batch_norm}
        constants.logger.info(
            f"Using batchnorm for rho cls: {fnn_args[constants.BATCH_NORM]}"
        )
        out_dim = 1  # For now, let us just predict the classifier confidence

        return FNN(
            in_dim=self.all_dim,
            out_dim=out_dim,  # number of rhos
            nn_arch=nn_arch,
            prefix=prefix,
            **fnn_args,
        )

    @abstractproperty
    def _model(self) -> nn.Module:
        raise NotImplementedError()

    @abstractproperty
    def _model_type(self):
        raise NotImplementedError()

    @property
    def _dir_path(self) -> Path:
        # TODO Change this to a better location
        return constants.PROJ_DIR / "results" / "models" / self._model_type

    def save_model(self, rec_model_name: str):
        """Saves the model inside the result folder

        Args:
            ver_model_name (str): _description_
        """
        self._dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {self._dir_path}")
        constants.logger.info(f"Saving model to {self._dir_path}")
        torch.save(self._model.state_dict(), self._dir_path / f"{rec_model_name}.pt")

    def load_model(self, *, rec_model_name: str):
        """Loads the model from the result folder

        Args:
            cls_model (nn.Module): _description_
            cls_model_name (str): _description_
        """
        print(
            f"Loading model from {str(self._dir_path.absolute())}/{rec_model_name}.pt"
        )
        constants.logger.info(f"Loading model from {self._dir_path}")
        self._model.load_state_dict(
            torch.load(
                self._dir_path / f"{rec_model_name}.pt", map_location=cu.get_device()
            )
        )

    def save_rec_probs(
        self, *, probs: np.ndarray, rec_model_name: str, dataset_split: str
    ):
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        assert dataset_split in [
            constants.TRNTST,
            constants.VAL,
            constants.TST,
        ], f"Invalid dataset split {dataset_split}"

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        (self._dir_path / f"{rec_model_name}").mkdir(parents=True, exist_ok=True)

        pkl.dump(
            probs,
            open(
                self._dir_path / f"{rec_model_name}" / f"recprobs-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved rec probs to {self._dir_path / f'{rec_model_name}' / f'recprobs-{dataset_split}.pkl'}"
        )

    def load_rec_probs(self, *, rec_model_name: str, dataset_split: str) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        assert dataset_split in [
            constants.TRN,
            constants.TRNTST,
            constants.VAL,
            constants.TST,
        ], f"Invalid dataset {dataset_split}"
        print(
            f"loaded rec probs from {self._dir_path / f'{rec_model_name}' / f'recprobs-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path / f"{rec_model_name}" / f"recprobs-{dataset_split}.pkl",
                "rb",
            )
        )

    def save_misc(
        self,
        *,
        object: np.ndarray,
        rec_model_name: str,
        object_name: str,
        dataset_split: str,
    ):
        """Saves the misc object to result/models/cls/<rec_model_name>/<object_name>-<dataset_split>.pkl

        Args:
            object (np.ndarray): _description_
            ver_model_name (str): _description_
            object_name (str): _description_
            dataset (str): _description_
        """
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        if isinstance(object, torch.Tensor):
            object = object.cpu().numpy()

        (self._dir_path / f"{rec_model_name}").mkdir(parents=True, exist_ok=True)

        if object_name.endswith(".pkl"):
            object_name = object_name[:-4]

        pkl.dump(
            object,
            open(
                self._dir_path
                / f"{rec_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved misc to {self._dir_path / f'{rec_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )

    def load_misc(
        self, *, rec_model_name: str, object_name: str, dataset_split: str
    ) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        print(
            f"loaded misc from {self._dir_path / f'{rec_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path
                / f"{rec_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "rb",
            )
        )


class TarnetRecModel(RecModel):
    def __init__(self, datasets, *args, **kwargs):
        super().__init__(datasets, *args, **kwargs)

        self.rec_beta_arms = nn.ModuleDict()
        for shift_id in range(self.num_shifts):
            self.rec_beta_arms[f"rec_beta_{shift_id}"] = self._rho_cls_model(
                prefix=f"rec_beta_{shift_id}", nn_arch=self.nn_arch, batch_norm=True
            )

    def forward(
        self,
        *,
        img: torch.Tensor,
        src_shift: list,
        rec_shift: list,
        sstar: torch.Tensor = None,
    ):
        """Forward for the TARNET recourse model

        Args:
            z (torch.Tensor):z for the current image
            beta (torch.Tensor): beta for the current image
            phi_x (torch.Tensor): embedding for the current image
            rec_beta_ids (torch.Tensor): the proposed beta recourse for the current image
        """

        src_shift_ids = [self.shift_idxs[entry] for entry in src_shift]
        src_shift_ids = torch.Tensor(src_shift_ids).long().to(cu.get_device())

        rec_shift_ids = [self.shift_idxs[entry] for entry in rec_shift]
        rec_shift_ids = torch.Tensor(rec_shift_ids).long().to(cu.get_device())

        shift_emb = self.shift_emb(src_shift_ids)
        if "x" in self.rec_input:
            img_emb = self.resnet_50(img)
            emb = torch.cat([img_emb, shift_emb], dim=1)
        if constants.SSTAR in self.rec_input:
            assert sstar is not None, "sstar is None"
            sstar_emb = self.sstar_fnn(sstar)
            emb = torch.cat([sstar_emb, shift_emb], dim=1)

        out = torch.zeros(len(emb)).to(cu.get_device(), dtype=torch.float32)
        for _ in range(self.num_shifts):
            idxs = torch.where(rec_shift_ids == _)[0]

            if len(idxs) == 1:
                tdu.batch_norm_off(self.rec_beta_arms[f"rec_beta_{_}"])

            if len(idxs) > 0:
                out[idxs] = (
                    out[idxs] + self.rec_beta_arms[f"rec_beta_{_}"](emb[idxs]).squeeze()
                )

            if len(idxs) == 1:
                tdu.batch_norm_on(self.rec_beta_arms[f"rec_beta_{_}"])

        out = self.sigmoid(out)
        return out

    def forward_all_beta(
        self,
        *,
        img: torch.Tensor,
        src_shift: list,
        sstar: torch.Tensor = None,
    ):
        """Forward for the TARNET recourse model for all the rec_beta_ids

        Args:
            z (torch.Tensor):z for the current image
            beta (torch.Tensor): beta for the current image
            phi_x (torch.Tensor): embedding for the current image
            rec_beta_ids (torch.Tensor): the proposed beta recourse for the current image
        """

        out = torch.zeros(len(src_shift), self.num_shifts).to(
            cu.get_device(), dtype=torch.float32
        )

        for idx, shift in enumerate(self.shifts):
            rec_shifts = [shift] * len(src_shift)
            out[:, idx] += self.forward(
                img=img, src_shift=src_shift, rec_shift=rec_shifts, sstar=sstar
            )

        out = self.sigmoid(out)
        return out

    @property
    def _model(self) -> nn.Module:
        return self

    @property
    def _model_type(self):
        return constants.TARNET_RECOURSE
