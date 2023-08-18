import torchvision
import constants as constants
import torch.nn as nn
import torch
from torchvision import models as tv_models
from src.data import data_helper as dh
from src.data import data_helper as dh
from src.data import dataset as ds
import numpy as np
from itertools import product
from pathlib import Path
from utils import common_utils as cu
from abc import ABC, abstractmethod, abstractproperty
import pickle as pkl
from utils import torch_data_utils as tdu


class ClsModel(nn.Module, ABC):
    def __init__(self, *, model, cls_dh: dh.ClsDataHelper, **kwargs) -> nn.Module:
        super(ClsModel, self).__init__()
        self.cls_dh = cls_dh
        self.model = model
        self.sm = nn.Softmax(dim=1)

    @property
    def _model(self) -> nn.Module:
        return self.model

    @abstractproperty
    def _model_type(self):
        raise NotImplementedError()

    @abstractproperty
    def _num_classes(self):
        raise NotImplementedError()

    @abstractmethod
    def forward_proba(self, input):
        out = self.model(input)
        return self.sm(out)

    @abstractmethod
    def forward_emb(self, input):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, input):
        return self.model(input)

    @abstractmethod
    def forward_labels(self, input):
        probs = self.forward_proba(input)
        probs, labels = torch.max(probs, dim=1)
        return labels

    @abstractproperty
    def _dir_path(self) -> Path:
        raise NotImplementedError()

    def save_model(self, cls_model_name: str):
        """Saves the model inside the result folder

        Args:
            cls_model (nn.Module): _description_
            cls_model_name (str): _description_
        """
        self._dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {self._dir_path}")
        constants.logger.info(f"Saving model to {self._dir_path}")
        torch.save(self._model.state_dict(), self._dir_path / f"{cls_model_name}.pt")

    def load_model(self, *, cls_model_name: str):
        """Loads the model from the result folder

        Args:
            cls_model (nn.Module): _description_
            cls_model_name (str): _description_
        """
        print(
            f"Loading model from {str(self._dir_path.absolute())}/{cls_model_name}.pt"
        )
        constants.logger.info(f"Loading model from {self._dir_path}")
        self._model.load_state_dict(
            torch.load(
                self._dir_path / f"{cls_model_name}.pt", map_location=cu.get_device()
            )
        )

    def save_cls_preds(
        self, *, preds: np.ndarray, cls_model_name: str, dataset_split: str
    ):
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        if dataset_split in [constants.TRN, constants.TRNTST]:
            cls_ds: ds.ClsDataset = self.cls_dh._trn_ds
            if cls_ds.load_cls_trn == True:
                dataset_split = f"{dataset_split}-real"
            else:
                dataset_split = f"{dataset_split}-rec"
        assert dataset_split in [
            constants.TRNTST,
            constants.VAL,
            constants.TST,
        ], f"Invalid dataset split {dataset_split}"

        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        (self._dir_path / f"{cls_model_name}").mkdir(parents=True, exist_ok=True)

        pkl.dump(
            preds,
            open(
                self._dir_path / f"{cls_model_name}" / f"clspreds-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved cls preds to {self._dir_path / f'{cls_model_name}' / f'clspreds-{dataset_split}.pkl'}"
        )

    def load_cls_preds(self, *, cls_model_name: str, dataset_split: str) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        assert dataset_split in [
            constants.TRN,
            constants.TRNTST,
            constants.VAL,
            constants.TST,
        ], f"Invalid dataset {dataset_split}"

        if dataset_split in [constants.TRN, constants.TRNTST]:
            cls_ds: ds.ClsDataset = self.cls_dh._trn_ds
            if cls_ds.load_cls_trn == True:
                dataset_split = f"{dataset_split}-real"
            else:
                dataset_split = f"{dataset_split}-rec"

        print(
            f"loaded cls preds from {self._dir_path / f'{cls_model_name}' / f'clspreds-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path / f"{cls_model_name}" / f"clspreds-{dataset_split}.pkl",
                "rb",
            )
        )

    def save_misc(
        self,
        *,
        object: np.ndarray,
        cls_model_name: str,
        object_name: str,
        dataset_split: str,
    ):
        """Saves the misc object to result/models/cls/<cls_model_name>/<object_name>-<dataset_split>.pkl

        Args:
            object (np.ndarray): _description_
            cls_model_name (str): _description_
            object_name (str): _description_
            dataset (str): _description_
        """
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        if dataset_split in [constants.TRN, constants.TRNTST]:
            cls_ds: ds.ClsDataset = self.cls_dh._trn_ds
            if cls_ds.load_cls_trn == True:
                object_name = f"{object_name}-real"
            else:
                object_name = f"{object_name}-rec"

        if isinstance(object, torch.Tensor):
            object = object.cpu().numpy()

        (self._dir_path / f"{cls_model_name}").mkdir(parents=True, exist_ok=True)

        if object_name.endswith(".pkl"):
            object_name = object_name[:-4]

        pkl.dump(
            object,
            open(
                self._dir_path
                / f"{cls_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved misc to {self._dir_path / f'{cls_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )

    def load_misc(
        self, *, cls_model_name: str, object_name: str, dataset_split: str
    ) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST

        if dataset_split in [constants.TRN, constants.TRNTST]:
            cls_ds: ds.ClsDataset = self.cls_dh._trn_ds
            if cls_ds.load_cls_trn == True:
                object_name = f"{object_name}-real"
            else:
                object_name = f"{object_name}-rec"
        if object_name.endswith(".pkl"):
            object_name = object_name[:-4]

        print(
            f"loaded misc from {self._dir_path / f'{cls_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path
                / f"{cls_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "rb",
            )
        )


class ResnetCls(ClsModel):
    def __init__(self, *, cls_dh: dh.ClsDataHelper, **kwargs) -> nn.Module:
        model = tv_models.resnet18(weights=None)
        print(f"Pretrn: False")
        constants.logger.info(f"Pretrn: False")
        self.num_classes = cls_dh._trn_ds._num_classes
        model.fc = FNN(
            in_dim=model.fc.in_features,
            out_dim=self.num_classes,
            nn_arch=[64],
            prefix="cls",
            **kwargs,
        )
        super().__init__(model=model, cls_dh=cls_dh, **kwargs)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    @property
    def _model_type(self):
        return constants.RESNET_CLS

    @property
    def _num_classes(self):
        return self.num_classes

    @property
    def _dir_path(self) -> Path:
        return constants.MODELDIR / "cls" / self.cls_dh._name / self._model_type

    def forward_proba(self, input):
        return super().forward_proba(input)

    def forward(self, input):
        return super().forward(input)

    def forward_labels(self, input):
        return super().forward_labels(input)

    def forward_emb(self, input):
        return self.feature_extractor(input).squeeze()


class ResnetZ(ResnetCls):
    """This model is the same as the classifier model!

    Args:
        ResnetCls (_type_): _description_
    """

    def __init__(self, *, cls_dh: dh.ClsDataHelper, **kwargs) -> nn.Module:
        super().__init__(cls_dh=cls_dh, **kwargs)
        self.z_out = kwargs.get(constants.Z_OUT, None)
        assert self.z_out in [constants.FINE_LABELS], f"Invalid z_out: {self.z_out}"
        trn_ds: ds.ClsDataset = self.cls_dh._trn_ds
        if self.z_out == constants.FINE_LABELS:
            self.num_classes = trn_ds._num_fine_classes
        else:
            raise ValueError(f"Invalid z_out: {self.z_out}")

    @property
    def _model_type(self):
        return constants.RESNET_Z

    @property
    def _num_classes(self) -> int:
        return self.num_classes

    @property
    def _dir_path(self) -> Path:
        return constants.MODELDIR / "Z" / self.cls_dh._name / self._model_type


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
            self.model.add_module(
                f"{self.prefix}-lReLU_{idx}", nn.LeakyReLU(inplace=True)
            )
            if self.batchnorm:
                self.model.add_module(f"{self.prefix}-bn_{idx}", nn.BatchNorm1d(hdim))

            prev = hdim
        self.model.add_module(f"{self.prefix}-last_layer", nn.Linear(prev, out_dim))

    def forward(self, input):
        return self.model(input)


class VerModel(nn.Module):
    # TODO: Make this as an abstract class if we need to test more architectures for verifier model
    """Creates a Verifier model with the following architecture:
    Accepts (z, \beta, z', \beta') as input
    Outuputs +1/-1 depending on if \beta' is more likely to act as recourse \beta.

    Optionally accepts embedding of $x$
    """

    def __init__(self, ver_dh: dh.VerDataHelper, *args, **kwargs):
        super().__init__()
        self.ver_dh = ver_dh

        trn_ds: ds.VerDataset = ver_dh._trn_ds
        self.zvals = trn_ds._z_vals
        self.betavals = trn_ds._beta_vals
        self.zstr = trn_ds._z_str
        self.betastr = trn_ds._beta_str

        self.nn_arch = kwargs.get(constants.NNARCH, [64, 32])
        self.use_imgemb = kwargs.get(constants.USE_IMGEMB, False)
        self.embdim = kwargs.get(constants.EMBDIM, 64)
        self.imgembdim = kwargs.get(constants.IMGEMBDIM, 512)  # 512 for resnet18

        self.ver_output = kwargs.get(constants.VER_OUTPUT, None)
        assert self.ver_output in [
            constants.ACCREJ,
            constants.DIFF_OF_DIFF,
        ], f"pass valid output type. Got {self.ver_output}"
        constants.logger.warning(f"Verifier model outputs: {self.ver_output}")
        print(f"Verifier model outputs: {self.ver_output}")

        self.z_arms = nn.ModuleDict()
        for idx, z in enumerate(self.zvals):
            assert z > 0, f"Invalid zdim {z}"
            self.z_arms[f"z_{idx}"] = Embedding(num_vocab=z, embdim=self.embdim)

        self.beta_arms = nn.ModuleDict()
        for idx, beta in enumerate(self.betavals):
            assert beta > 0, f"Invalid betadim {beta}"
            self.beta_arms[f"beta_{idx}"] = Embedding(
                num_vocab=beta, embdim=self.embdim
            )

        self.zdim = len(self.zvals) * self.embdim
        self.betadim = len(self.betavals) * self.embdim

        self.all_dim = self.zdim + self.betadim
        self.all_dim = 2 * self.all_dim

        if self.use_imgemb:
            self.all_dim += self.imgembdim

        del kwargs[constants.NNARCH]
        out_dim = (
            3
            if self.ver_output == constants.ACCREJ
            else 1
            if self.ver_output == constants.DIFF_OF_DIFF
            else None
        )
        constants.logger.info(f"Verifier output dim: {out_dim}")
        self.verifier = FNN(
            in_dim=self.all_dim,
            out_dim=out_dim,
            nn_arch=self.nn_arch,
            prefix=constants.VER,
            *args,
            **kwargs,
        )

        self.sm = nn.Softmax(dim=1)

    def forward(
        self,
        z: torch.Tensor,
        beta: torch.Tensor,
        zprime: torch.Tensor,
        betaprime: torch.Tensor,
        phi_x: torch.Tensor = None,
    ):
        if z.ndim == 1:
            z = z.unsqueeze(1)
        if beta.ndim == 1:
            beta = beta.unsqueeze(1)
        if zprime.ndim == 1:
            zprime = zprime.unsqueeze(1)
        if betaprime.ndim == 1:
            betaprime = betaprime.unsqueeze(1)

        zemb = torch.cat(
            [
                self.z_arms[f"z_{idx}"](z[:, idx].view(-1, 1))
                for idx in range(len(self.zvals))
            ],
            dim=1,
        )
        betaemb = torch.cat(
            [
                self.beta_arms[f"beta_{idx}"](beta[:, idx].view(-1, 1))
                for idx in range(len(self.betavals))
            ],
            dim=1,
        )
        zpemb = torch.cat(
            [
                self.z_arms[f"z_{idx}"](zprime[:, idx].view(-1, 1))
                for idx in range(len(self.zvals))
            ],
            dim=1,
        )
        bpemb = torch.cat(
            [
                self.beta_arms[f"beta_{idx}"](betaprime[:, idx].view(-1, 1))
                for idx in range(len(self.betavals))
            ],
            dim=1,
        )
        emb = torch.cat([zemb, betaemb, zpemb, bpemb], dim=1)
        if self.use_imgemb:
            emb = torch.cat([emb, phi_x], dim=1)

        return self.verifier(emb)

    def forward_proba(self, z, beta, zprime, betaprime, phi_x=None):
        """Forwards and finds the probability of the output

        Args:
            z (_type_): _description_
            beta (_type_): _description_
            zprime (_type_): _description_
            betaprime (_type_): _description_
            phi_x (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.ver_output == constants.ACCREJ:
            return self.sm(self.forward(z, beta, zprime, betaprime, phi_x))
        elif self.ver_output == constants.DIFF_OF_DIFF:
            return self.forward(z, beta, zprime, betaprime, phi_x).squeeze()
        else:
            raise ValueError(f"ver_output = {self.ver_output} is not supported")

    @property
    def _model(self) -> nn.Module:
        return self

    @abstractproperty
    def _model_type(self):
        return constants.FNNVER

    @property
    def _num_classes(self):
        if self.ver_output == constants.ACCREJ:
            return 3
        else:
            raise ValueError(
                f"Why are u asking num classes for verifier that emits {self.ver_output}"
            )

    def forward_labels(self, z, beta, zprime, betaprime, phi_x=None):
        """Returns labels \in {-1, 0, +1}

        Args:
            z (_type_): _description_
            beta (_type_): _description_
            zprime (_type_): _description_
            betaprime (_type_): _description_
            phi_x (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.ver_output == constants.ACCREJ:
            probs = self.forward_proba(z, beta, zprime, betaprime, phi_x=phi_x)
            probs, labels = torch.max(probs, dim=1)
            return labels - 1
        elif self.ver_output == constants.DIFF_OF_DIFF:
            return self.forward(z, beta, zprime, betaprime, phi_x=phi_x).squeeze()
        else:
            raise ValueError(f"ver_output = {self.ver_output} is not supported")

    @property
    def _dir_path(self) -> Path:
        return constants.MODELDIR / constants.VER / self.ver_dh._name / self._model_type

    def save_model(self, ver_model_name: str):
        """Saves the model inside the result folder

        Args:
            ver_model_name (str): _description_
        """
        self._dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {self._dir_path}")
        constants.logger.info(f"Saving model to {self._dir_path}")
        torch.save(self.state_dict(), self._dir_path / f"{ver_model_name}.pt")

    def load_model(self, *, ver_model_name: str):
        """Loads the model from the result folder

        Args:
            cls_model (nn.Module): _description_
            cls_model_name (str): _description_
        """
        print(
            f"Loading model from {str(self._dir_path.absolute())}/{ver_model_name}.pt"
        )
        constants.logger.info(f"Loading model from {self._dir_path}")
        self._model.load_state_dict(
            torch.load(
                self._dir_path / f"{ver_model_name}.pt", map_location=cu.get_device()
            )
        )
        self._model.to(cu.get_device())

    def save_ver_probs(
        self, *, probs: np.ndarray, ver_model_name: str, dataset_split: str
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

        (self._dir_path / f"{ver_model_name}").mkdir(parents=True, exist_ok=True)

        pkl.dump(
            probs,
            open(
                self._dir_path / f"{ver_model_name}" / f"verprobs-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved ver probs to {self._dir_path / f'{ver_model_name}' / f'verprobs-{dataset_split}.pkl'}"
        )

    def load_ver_probs(self, *, ver_model_name: str, dataset_split: str) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        assert dataset_split in [
            constants.TRN,
            constants.TRNTST,
            constants.VAL,
            constants.TST,
        ], f"Invalid dataset {dataset_split}"
        print(
            f"loaded cls probs from {self._dir_path / f'{ver_model_name}' / f'verprobs-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path / f"{ver_model_name}" / f"verprobs-{dataset_split}.pkl",
                "rb",
            )
        )

    def save_misc(
        self,
        *,
        object: np.ndarray,
        ver_model_name: str,
        object_name: str,
        dataset_split: str,
    ):
        """Saves the misc object

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

        (self._dir_path / f"{ver_model_name}").mkdir(parents=True, exist_ok=True)

        if object_name.endswith(".pkl"):
            object_name = object_name[:-4]

        pkl.dump(
            object,
            open(
                self._dir_path
                / f"{ver_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "wb",
            ),
        )
        print(
            f"Saved misc to {self._dir_path / f'{ver_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )

    def load_misc(
        self, *, ver_model_name: str, object_name: str, dataset_split: str
    ) -> np.ndarray:
        if dataset_split == constants.TRN:
            dataset_split = constants.TRNTST
        print(
            f"loaded misc from {self._dir_path / f'{ver_model_name}' / f'{object_name}-{dataset_split}.pkl'}"
        )
        return pkl.load(
            open(
                self._dir_path
                / f"{ver_model_name}"
                / f"{object_name}-{dataset_split}.pkl",
                "rb",
            )
        )


class RecModel(nn.Module, ABC):
    """This is an abstract class for recourse models.
    For now, we would want to try the TARNET based model.
    """

    def __init__(
        self, cls_dh: dh.ClsDataHelper, ver_dh: dh.VerDataHelper, *args, **kwargs
    ):
        super().__init__()
        self.cls_dh = cls_dh
        self.ver_dh = ver_dh

        vertrn_ds: ds.VerDataset = ver_dh._trn_ds
        self.zvals = vertrn_ds._z_vals
        self.betavals = vertrn_ds._beta_vals
        self.zstr = vertrn_ds._z_str
        self.betastr = vertrn_ds._beta_str

        self.num_betas = vertrn_ds._num_betas

        self.nn_arch = kwargs.get(constants.NNARCH, [64, 32])
        self.embdim = kwargs.get(constants.EMBDIM, 64)
        self.imgembdim = kwargs.get(constants.IMGEMBDIM, 512)  # 512 for resnet18

        self.z_arms = nn.ModuleDict()
        for idx, z in enumerate(self.zvals):
            assert z > 0, f"Invalid zdim {z}"
            self.z_arms[f"z_{idx}"] = Embedding(num_vocab=z, embdim=self.embdim)

        self.beta_arms = nn.ModuleDict()
        for idx, beta in enumerate(self.betavals):
            assert beta > 0, f"Invalid betadim {beta}"
            self.beta_arms[f"beta_{idx}"] = Embedding(
                num_vocab=beta, embdim=self.embdim
            )

        self.zdim = len(self.zvals) * self.embdim
        self.betadim = len(self.betavals) * self.embdim

        self.all_dim = self.zdim + self.betadim

        self.all_dim = self.all_dim + self.imgembdim

        self.sm = nn.Softmax(dim=1)

    def zbeta_phix_forward(
        self,
        z: torch.Tensor,
        beta: torch.Tensor,
        phi_x: torch.Tensor = None,
    ):
        if z.ndim == 1:
            z = z.unsqueeze(1)
        if beta.ndim == 1:
            beta = beta.unsqueeze(1)

        zemb = torch.cat(
            [
                self.z_arms[f"z_{idx}"](z[:, idx].view(-1, 1))
                for idx in range(len(self.zvals))
            ],
            dim=1,
        )
        betaemb = torch.cat(
            [
                self.beta_arms[f"beta_{idx}"](beta[:, idx].view(-1, 1))
                for idx in range(len(self.betavals))
            ],
            dim=1,
        )

        zbeta_phix_emb = torch.cat([zemb, betaemb, phi_x], dim=1)

        return zbeta_phix_emb

    @abstractmethod
    def forward(
        self,
        *,
        z: torch.Tensor,
        beta: torch.Tensor,
        phi_x: torch.Tensor,
        rec_beta_ids: torch.Tensor,
    ):
        raise NotImplementedError()

    @abstractmethod
    def forward_all_beta(
        self,
        *,
        z: torch.Tensor,
        beta: torch.Tensor,
        phi_x: torch.Tensor,
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
        return constants.MODELDIR / constants.REC / self.cls_dh._name / self._model_type

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
    def __init__(
        self, cls_dh: dh.ClsDataHelper, ver_dh: dh.VerDataHelper, *args, **kwargs
    ):
        super().__init__(cls_dh, ver_dh, *args, **kwargs)

        self.rec_beta_arms = nn.ModuleDict()
        for bid in range(self.num_betas):
            self.rec_beta_arms[f"rec_beta_{bid}"] = self._rho_cls_model(
                prefix=f"rec_beta_{bid}", nn_arch=self.nn_arch, batch_norm=True
            )

    def forward(
        self,
        *,
        z: torch.Tensor,
        beta: torch.Tensor,
        phi_x: torch.Tensor,
        rec_beta_ids: torch.Tensor,
    ):
        """Forward for the TARNET recourse model

        Args:
            z (torch.Tensor):z for the current image
            beta (torch.Tensor): beta for the current image
            phi_x (torch.Tensor): embedding for the current image
            rec_beta_ids (torch.Tensor): the proposed beta recourse for the current image
        """
        z_beta_emb = self.zbeta_phix_forward(z=z, beta=beta, phi_x=phi_x)

        out = torch.zeros(len(phi_x)).to(cu.get_device(), dtype=torch.float64)
        for _ in range(self.num_betas):
            idxs = torch.where(rec_beta_ids == _)[0]

            if len(idxs) == 1:
                tdu.batch_norm_off(self.rec_beta_arms[f"rec_beta_{_}"])

            if len(idxs) > 0:
                out[idxs] = (
                    out[idxs]
                    + self.rec_beta_arms[f"rec_beta_{_}"](z_beta_emb[idxs]).squeeze()
                )

            if len(idxs) == 1:
                tdu.batch_norm_on(self.rec_beta_arms[f"rec_beta_{_}"])

        return out

    def forward_all_beta(
        self,
        *,
        z: torch.Tensor,
        beta: torch.Tensor,
        phi_x: torch.Tensor,
    ):
        """Forward for the TARNET recourse model for all the rec_beta_ids

        Args:
            z (torch.Tensor):z for the current image
            beta (torch.Tensor): beta for the current image
            phi_x (torch.Tensor): embedding for the current image
            rec_beta_ids (torch.Tensor): the proposed beta recourse for the current image
        """
        out = torch.zeros(len(phi_x), self.num_betas).to(
            cu.get_device(), dtype=torch.float64
        )
        for bid in range(self.num_betas):
            rec_beta_ids = (
                torch.ones(len(phi_x)).to(cu.get_device(), dtype=torch.long) * bid
            )
            rec_beta_ids = rec_beta_ids.view(-1, 1)
            out[:, bid] += self.forward(
                z=z, beta=beta, phi_x=phi_x, rec_beta_ids=rec_beta_ids
            )
        return out

    @property
    def _model(self) -> nn.Module:
        return self

    @property
    def _model_type(self):
        return constants.TARNET_RECOURSE
