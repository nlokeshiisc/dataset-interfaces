import torch
import torch.nn as nn
from src import models
from src import data_helper as dh
import constants
from tqdm import tqdm
from utils import common_utils as cu
from torch.utils.data import DataLoader
from utils import torch_data_utils as tdu
import numpy as np
from src import dataset as ds
from tqdm import tqdm
from utils import torch_data_utils as tdu
from dataset_interfaces import utils as dsi_utils
from dataset_interfaces import imagenet_utils as dsi_imutils
from utils import torch_utils as tu


class RecHelper:
    def __init__(
        self,
        rec_model: models.RecModel,
        rec_dh: dh.RecDataHelper,
        rec_model_name: str,
        **kwargs,
    ):
        self.rec_model = rec_model
        self.rec_dh = rec_dh
        self.rec_model_name = rec_model_name
        self.num_shifts = self.rec_model.num_shifts

        self.sel_sysnets = kwargs.get(constants.SEL_SYSNETS)
        self.sel_sysnets = sorted(self.sel_sysnets)
        self.cls_to_imgnet = {}
        for _, c in enumerate(self.sel_sysnets):
            self.cls_to_imgnet[_] = dsi_imutils.sysnet_to_clsid[c]

        constants.logger.info(
            f"seld_classes: {self.sel_sysnets}, imagenet_ids: {self.cls_to_imgnet}"
        )
        print(f"seld_classes: {self.sel_sysnets}, imagenet_ids: {self.cls_to_imgnet}")

        self.lr = kwargs.get(constants.LRN_RATE, 1e-4)
        self.checkpoints = kwargs.get(constants.CHECKPOINTS, [])
        self.ctr_lambda = kwargs.get(constants.CTR_LAMBDA, 0.1)
        self.enforce_ctrloss = kwargs.get(constants.ENFORCE_CTRLOSS, False)
        self.rec_input = kwargs.get(constants.INPUT)

        constants.logger.info(f"rec_model_name: {self.rec_model_name}")
        print(f"rec_model_name: {self.rec_model_name}")

    def __str__(self) -> str:
        return f"{self.rec_model_name}"

    @property
    def _num_shifts(self) -> int:
        return self.num_shifts

    @property
    def _xent_loss(self):
        return nn.CrossEntropyLoss(reduction="mean")

    @property
    def _mse_loss(self):
        return nn.MSELoss(reduction="mean")

    @property
    def _mae_loss(self):
        return nn.L1Loss(reduction="mean")

    @property
    def _margin_loss(self):
        return nn.MarginRankingLoss(reduction="mean", margin=self.margin)

    @property
    def _margin_loss_perex(self):
        return nn.MarginRankingLoss(reduction="none", margin=self.margin)

    @property
    def _model(self) -> models.RecModel:
        return self.rec_model

    def load_model(
        self,
    ):
        self._model.load_model(rec_model_name=self.rec_model_name)

    def save_model(self):
        self._model.save_model(rec_model_name=self.rec_model_name)

    def train_rec(self, **kwargs):
        """Trains the ver model on the real dataset"""
        epochs = kwargs.get(constants.EPOCHS, 20)

        trn_loader = self.rec_dh._trn_loader

        itr = 1
        global_step = 1

        self._model.train()

        self._model.to(cu.get_device(), dtype=torch.float32)
        params = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        for epoch in range(epochs):
            self._model.train()

            pbar = tqdm(trn_loader, total=len(trn_loader))
            for idxs, batch in pbar:
                # Half precision for training screws up training: https://github.com/soumith/cudnn.torch/issues/377
                imgs = batch[constants.IMAGE].to(cu.get_device(), dtype=torch.float32)
                labels = batch[constants.LABEL].to(cu.get_device(), dtype=torch.long)
                rhos = batch[constants.RHO].to(cu.get_device(), dtype=torch.float32)
                cls_loss = batch[constants.LOSS].to(
                    cu.get_device(), dtype=torch.float32
                )
                src_shifts = batch[constants.SHIFT]
                sstar = None
                if constants.SSTAR in self.rec_input:
                    sstar = batch[constants.SSTAR].to(
                        cu.get_device(), dtype=torch.float32
                    )

                if len(idxs) == 1:
                    tdu.batch_norm_off(self._model)

                # For Factual model, src shift is the same as rec shift while training
                rho_preds = self._model.forward(
                    img=imgs, src_shift=src_shifts, rec_shift=src_shifts, sstar=sstar
                )

                fct_loss = self._mse_loss(rho_preds.squeeze(), rhos.squeeze())
                loss = fct_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if len(idxs) == 1:
                    tdu.batch_norm_on(self._model)

                pbar.set_postfix({"fct_loss": fct_loss.item()})

                itr += 1
                global_step += 1

                if constants.sw is not None:
                    constants.sw.add_scalar(
                        "train/fct_loss", fct_loss.item(), global_step
                    )

            if epoch % 5 == 0:
                # TODO Add testing code
                pass

            # update the learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

    @torch.inference_mode()
    def get_rec_proba(
        self, save_probs=False, dataset_split=constants.TST
    ) -> torch.Tensor:
        """Saves the recourse probabilities for all the betas in the dataset.
        The probs are saved in beta_ids order.

        Args:
            save_proba (bool, optional): _description_. Defaults to False.
            dataset_split (_type_, optional): _description_. Defaults to constants.TST.
        """
        raise NotImplementedError()
        assert (
            dataset_split == constants.TST
        ), "we perform inference only on the test dataset"

        try:
            rec_probs = self.rec_model.load_rec_probs(
                rec_model_name=self.rec_model_name, dataset_split=dataset_split
            )
            return torch.Tensor(rec_probs)
        except:
            print("Computing rec probs")

        loader = self.rec_dh._tst_loader
        self._model.eval()
        self._model.to(cu.get_device())

        rec_probs = []
        with torch.no_grad():
            for idxs, img_dict in loader:
                z = img_dict["z"].to(cu.get_device())
                beta = img_dict[constants.BETA].to(cu.get_device())
                img_emb = img_dict[constants.IMGEMB].to(cu.get_device())

                all_rho_preds = self._model.forward_all_beta(
                    z=z,
                    beta=beta,
                    phi_x=img_emb,
                )
                rec_probs.append(all_rho_preds)

        rec_probs = torch.cat(rec_probs, dim=0)

        if save_probs == True:
            self.rec_model.save_rec_probs(
                probs=rec_probs,
                rec_model_name=self.rec_model_name,
                dataset_split=dataset_split,
            )

        assert len(rec_probs) == len(loader.dataset)
        return rec_probs

    def get_rec_labels(
        self, save_probs=False, dataset_split=constants.TST
    ) -> torch.Tensor:
        """Computes the recourse labels for the real tst dataset

        Args:
            save_probs (bool, optional): _description_. Defaults to False.
            dataset_split (_type_, optional): _description_. Defaults to constants.TST.
        """
        raise NotImplementedError()
        assert dataset_split == constants.TST
        rec_probs = self.get_rec_proba(
            save_probs=save_probs, dataset_split=dataset_split
        )
        rec_conf, rec_labels = torch.max(rec_probs, dim=1)
        return rec_labels.detach().cpu()

    @torch.inference_mode()
    def evaluate_rec(self, save_probs=False, dataset_split=constants.TST) -> float:
        """Computes the accuracy at 100% recourse for the real test dataset

        Args:
            save_probs (bool, optional): _description_. Defaults to False.
            dataset_split (_type_, optional): _description_. Defaults to constants.TST.
        """
        assert dataset_split == constants.TST, "only test dataset shall be recoursed"

        if dataset_split == constants.TST:
            loader = self.rec_dh._tst_loader

        acc_meter = tu.AccuracyMeter(track=["acc"])
        num_beta = self._num_shifts

        self.rec_model.eval()
        self.rec_model.to(cu.get_device(), dtype=torch.float32)

        pbar = tqdm(loader, total=len(loader))
        rho_preds = []
        rhos = []
        with torch.no_grad():
            for idx, batch in pbar:
                img = batch[constants.IMAGE].to(cu.get_device(), dtype=torch.float32)
                label = batch[constants.LABEL].to(cu.get_device(), dtype=torch.float32)
                rho = batch[constants.RHO].to(cu.get_device(), dtype=torch.float32)
                src_shift = batch[constants.SHIFT]

                sstar = None
                if constants.SSTAR in self.rec_input:
                    sstar = batch[constants.SSTAR].to(
                        cu.get_device(), dtype=torch.float32
                    )

                rho_pred = self._model.forward_all_beta(
                    img=img, src_shift=src_shift, sstar=sstar
                )
                rho_preds.append(rho_pred)
                rhos.append(rho)

        # For each z, b, b' what is the predicted rho
        rho_preds = torch.cat(rho_preds, dim=0).view(-1, num_beta)

        # For each z \times \cal{B} what is the true rho
        rhos = torch.cat(rhos, dim=0).view(-1, num_beta)

        # Repeat the true rho num_beta times to mimic the tru z, b, b' structure
        rhos = rhos.repeat_interleave(num_beta, dim=0)

        # Take the argmax shift
        pred_max_shift = torch.argmax(rho_preds, dim=1)

        # For the argmax shift, what is the true rho
        rec_cnf = torch.gather(rhos, 1, pred_max_shift.view(-1, 1)).squeeze()

        # If rec were optimal, what is the true rho
        max_cnf, _ = torch.max(rhos, dim=1)

        print(
            f"Before Rec: {torch.mean(rhos).item()}, After Rec: {torch.mean(rec_cnf).item()}, max possible: {torch.mean(max_cnf).item()}"
        )
        print(f"Regret: {torch.mean(torch.abs(rec_cnf - max_cnf)).item()}")
        pass
