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

        self.lr = kwargs.get(constants.LRN_RATE, 0.005)
        self.checkpoints = kwargs.get(constants.CHECKPOINTS, [])
        self.ctr_lambda = kwargs.get(constants.CTR_LAMBDA, 0.1)
        self.enforce_ctrloss = kwargs.get(constants.ENFORCE_CTRLOSS, False)

        constants.logger.info(f"rec_model_name: {self.rec_model_name}")
        print(f"rec_model_name: {self.rec_model_name}")

    def __str__(self) -> str:
        return f"{self.rec_model_name}"

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

        self._model.to(cu.get_device(), dtype=torch.float16)
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

            log_ver_pc = []

            pbar = tqdm(trn_loader, total=len(trn_loader))
            for idxs, imgs, ys, shifts, rhos in pbar:
                imgs = imgs.to(cu.get_device(), dtype=torch.float16)
                ys = ys.to(cu.get_device(), dtype=torch.long)
                rhos = rhos.to(cu.get_device(), dtype=torch.float16)

                if len(idxs) == 1:
                    tdu.batch_norm_off(self._model)

                rho_preds = self._model.forward(img=imgs, src_shift=shifts)
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
        assert dataset_split == constants.TST
        rec_probs = self.get_rec_proba(
            save_probs=save_probs, dataset_split=dataset_split
        )
        rec_conf, rec_labels = torch.max(rec_probs, dim=1)
        return rec_labels.detach().cpu()

    def rec_acc(self, save_probs=False, dataset_split=constants.TST) -> float:
        """Computes the accuracy at 100% recourse for the real test dataset

        Args:
            save_probs (bool, optional): _description_. Defaults to False.
            dataset_split (_type_, optional): _description_. Defaults to constants.TST.
        """
        assert dataset_split == constants.TST, "only test dataset shall be recoursed"
        real_pred_labels = self.cls_helper.predict_labels(
            loader=self.cls_dh._tst_loader,
            misc_name=constants.REAL_PREDS,
            dataset_split=constants.TST,
            save_probs=True,
        )
        cls_tst_ds: ds.ClsDataset = self.cls_dh._tst_ds
        num_beta = cls_tst_ds._num_betas

        real_pred_labels = real_pred_labels.view(-1, num_beta)
        real_pred_labels = torch.repeat_interleave(real_pred_labels, num_beta, dim=0)

        real_gt_labels = torch.Tensor(cls_tst_ds._image_labels).long()

        rec_beta_ids = self.get_rec_labels(
            save_probs=True, dataset_split=dataset_split
        ).view(-1, 1)

        real_pred_labels = torch.gather(real_pred_labels, dim=1, index=rec_beta_ids)

        rec_acc = torch.sum(
            real_pred_labels.squeeze() == real_gt_labels.squeeze()
        ).item() / len(real_pred_labels)

        return rec_acc
