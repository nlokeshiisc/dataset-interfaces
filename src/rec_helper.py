import torch
import torch.nn as nn
from src import models
from src.data import data_helper as dh
import constants
from tqdm import tqdm
from utils import common_utils as cu
from torch.utils.data import DataLoader
from utils import torch_data_utils as tdu
import numpy as np
from src.data import dataset as ds
from tqdm import tqdm
from src import ver_helper as verh
from src import cls_helper as clsh
from utils import torch_data_utils as tdu


class RecHelper:
    def __init__(
        self,
        rec_model: models.RecModel,
        rec_dh: dh.RecDataHelper,
        rec_model_name: str,
        ver_dh: dh.VerDataHelper = None,
        ver_helper: verh.VerHelper = None,
        cls_dh: dh.ClsDataHelper = None,
        cls_helper: clsh.ClsHelper = None,
        **kwargs,
    ):
        self.rec_model = rec_model
        self.rec_dh = rec_dh
        self.rec_model_name = rec_model_name
        self.ver_dh: dh.VerDataHelper = ver_dh
        self.ver_helper: verh.VerHelper = ver_helper
        self.cls_dh: dh.ClsDataHelper = cls_dh
        self.cls_helper: clsh.ClsHelper = cls_helper

        self.lr = kwargs.get(constants.LRN_RATE, 0.005)
        self.checkpoints = kwargs.get(constants.CHECKPOINTS, [])
        self.margin = kwargs.get(constants.MARGIN, 0.2)
        self.ctr_lambda = kwargs.get(constants.CTR_LAMBDA, 0.1)
        self.use_verifier = kwargs.get(constants.USE_VER, False)
        self.enforce_ctrloss = kwargs.get(constants.ENFORCE_CTRLOSS, False)

        if self.use_verifier == True:
            assert (
                self.ver_dh is not None and self.ver_helper is not None
            ), "If u want to use the the verifier in ctr training, pass both the ver dh and ver helper."

        assert not (
            self.use_verifier and not self.enforce_ctrloss
        ), "When we use verifier, it makes sense only when ctr loss is imposed."

        if "ctr" not in self.rec_model_name and self.enforce_ctrloss == True:
            self.rec_model_name = f"{self.rec_model_name}-ctr={self.ctr_lambda}"
        if "ver" not in self.rec_model_name and self.use_verifier == True:
            self.rec_model_name = f"{self.rec_model_name}-ver"

        constants.logger.info(f"rec_model_name: {self.rec_model_name}")
        print(f"rec_model_name: {self.rec_model_name}")

    def __str__(self) -> str:
        name = f"{constants.REC}-{self.rec_model_name}"
        if self.enforce_ctrloss:
            name = f"{name}-ctr={self.ctr_lambda}"
            name = f"{name}-margin={self.margin}"
            if self.use_verifier:
                name = f"{name}-ver"
            else:
                name = f"{name}-no_ver"
        return name

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

        self._model.to(cu.get_device(), dtype=torch.float64)
        params = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        trn_ds: ds.VerDataset = self.rec_dh._trn_ds
        sim_ds: ds.VerDataset = self.rec_dh._sim_ds

        collect_rho = (
            lambda dset, z, b: torch.FloatTensor(
                [dset.z_beta_rho(zz.item(), bb.tolist()) for zz, bb in zip(z, b)]
            )
            .view(-1, 1)
            .to(cu.get_device(), dtype=torch.float64)
        )

        for epoch in range(epochs):
            self._model.train()

            log_ver_pc = []

            pbar = tqdm(trn_loader, total=len(trn_loader))
            for idxs, img_dict in pbar:
                z = img_dict["z"].to(cu.get_device())
                beta = img_dict[constants.BETA].to(cu.get_device())
                fct_betaids = img_dict[constants.BETAID].to(cu.get_device())

                rho = img_dict[constants.RHO].to(cu.get_device(), dtype=torch.float64)
                img_emb = img_dict[constants.IMGEMB].to(cu.get_device())

                if len(idxs) == 1:
                    tdu.batch_norm_off(self._model)

                fct_rho_preds = self._model.forward(
                    z=z, beta=beta, phi_x=img_emb, rec_beta_ids=fct_betaids
                )

                fct_loss = self._mse_loss(fct_rho_preds.squeeze(), rho.squeeze())

                loss = fct_loss

                if self.enforce_ctrloss == True:
                    rec_beta = torch.cat([trn_ds.sample_beta() for _ in z]).to(
                        cu.get_device()
                    )

                    if self.rec_dh._dataset_name == constants.SHAPENET:
                        rec_beta_ids = [
                            constants.shapenet_beta_to_idx_dict[
                                constants.betatostr_fn(rb.cpu().tolist())
                            ]
                            for rb in rec_beta
                        ]
                    elif self.rec_dh._dataset_name == constants.MEDICAL:
                        rec_beta_ids = [
                            constants.medical_beta_to_idx_dict[
                                constants.betatostr_fn(rb.cpu().tolist())
                            ]
                            for rb in rec_beta
                        ]
                    else:
                        raise ValueError(
                            f"Unknown dataset name {self.rec_dh._dataset_name}"
                        )
                    rec_beta_ids = (
                        torch.LongTensor(rec_beta_ids).view(-1, 1).to(cu.get_device())
                    )

                    sim_rho = collect_rho(sim_ds, z, beta)
                    sim_rho_rec = collect_rho(sim_ds, z, rec_beta)

                    rec_labels = torch.zeros(len(sim_rho), 1).to(cu.get_device())
                    rec_better = (sim_rho_rec > (sim_rho + self.margin)).to(
                        cu.get_device()
                    )
                    rec_worse = (sim_rho_rec < (sim_rho - self.margin)).to(
                        cu.get_device()
                    )
                    rec_labels[rec_better] = 1
                    rec_labels[rec_worse] = -1

                    rank_loss_idxs = torch.where(rec_labels != 0)[0]
                    mse_loss_idxs = torch.where(rec_labels == 0)[0]

                    """
                    Notes of marginRankingloss:
                    If y=1, then it assumed the first input should be ranked higher 
                    (have a larger value) than the second input, and vice-versa for y = −1
                    """
                    rec_rho_preds = self._model.forward(
                        z=z, beta=beta, phi_x=img_emb, rec_beta_ids=rec_beta_ids
                    )

                    ctr_loss = torch.zeros(len(rec_rho_preds)).to(
                        cu.get_device(), dtype=torch.float64
                    )
                    if len(rank_loss_idxs) > 0:
                        ctr_rank_loss = self._margin_loss_perex(
                            rec_rho_preds[rank_loss_idxs].squeeze(),
                            fct_rho_preds[rank_loss_idxs].squeeze(),
                            rec_labels[rank_loss_idxs].squeeze(),
                        )
                        ctr_loss[rank_loss_idxs] = ctr_rank_loss

                    if len(mse_loss_idxs) > 0:
                        ctr_mse_loss = self._mse_loss(
                            rec_rho_preds[mse_loss_idxs].squeeze(),
                            fct_rho_preds[mse_loss_idxs].squeeze(),
                        )
                        ctr_loss[mse_loss_idxs] = ctr_mse_loss

                    if self.use_verifier == False:
                        ctr_loss = torch.mean(ctr_loss)
                    else:
                        ver_labels = self.ver_helper.predict_labels_batch(
                            z=z, beta=beta, zp=z, betap=rec_beta, phi_x=img_emb
                        )
                        if self.ver_helper._ver_output == constants.ACCREJ:
                            ctr_ver = (ver_labels == 1).to(cu.get_device())
                        elif self.ver_helper._ver_output == constants.DIFF_OF_DIFF:
                            ctr_ver = (
                                torch.abs(ver_labels)
                                < self.ver_helper._diff_diff_crct_margin
                            ).to(cu.get_device())
                        else:
                            raise ValueError(
                                f"Unknown ver output type {self.ver_helper._ver_output}"
                            )
                        num_verified = torch.sum(ctr_ver).item() / len(idxs)
                        log_ver_pc.append(num_verified)

                        if num_verified > 0:
                            if len(idxs) == 1:
                                ctr_loss = ctr_loss.squeeze()
                            else:
                                ctr_loss = torch.mean(ctr_loss[ctr_ver.squeeze()])
                        else:
                            ctr_loss = torch.tensor(0.0).to(cu.get_device())

                    loss = loss + (self.ctr_lambda * ctr_loss)

                if torch.isnan(loss):
                    pass

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if len(idxs) == 1:
                    tdu.batch_norm_on(self._model)

                if constants.sw is not None:
                    constants.sw.add_scalar(
                        tag="fct_loss",
                        scalar_value=fct_loss.item(),
                        global_step=global_step,
                    )
                    if self.enforce_ctrloss:
                        constants.sw.add_scalar(
                            tag="ctr_loss",
                            scalar_value=ctr_loss.item(),
                            global_step=global_step,
                        )
                        constants.sw.add_scalar(
                            tag="tot_loss",
                            scalar_value=loss.item(),
                            global_step=global_step,
                        )
                        if self.use_verifier == True:
                            constants.sw.add_scalar(
                                tag="num_verified",
                                scalar_value=num_verified,
                                global_step=global_step,
                            )

                if self.enforce_ctrloss:
                    if self.use_verifier:
                        pbar.set_postfix(
                            {
                                "floss": fct_loss.item(),
                                "ctrloss": ctr_loss.item(),
                                "ver_pc": num_verified,
                            }
                        )
                    else:
                        pbar.set_postfix(
                            {"floss": fct_loss.item(), "ctrloss": ctr_loss.item()}
                        )
                else:
                    pbar.set_postfix({"fct_loss": fct_loss.item()})

                itr += 1
                global_step += 1

            if self.checkpoints is not None and (epoch + 1) in self.checkpoints:
                self.rec_model.save_model(
                    ver_model_name=f"{self.rec_model_name}-{epoch+1}"
                )

            if self.use_verifier and self.enforce_ctrloss:
                constants.sw.add_scalar(
                    tag="ver_pc",
                    scalar_value=np.mean(log_ver_pc),
                    global_step=epoch + 1,
                )
                constants.logger.info(
                    f"ver_pc: {np.mean(log_ver_pc)} at epoch: {epoch+1}"
                )

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
