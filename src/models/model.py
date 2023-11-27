import csv
import os
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric

from src.utils import (
    get_dist_info,
    is_master,
    reduce_value,
)
from src.utils.vis_utils import (
    save_depth_as_uint16png_upload,
    save_image,
    merge_into_row,
    batch_save,
    batch_save_kitti,
    padding_kitti,
)


class DepthLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metric,
        monitor,
        save_dir,
        base_lr,
        dataset,
        is_warmup=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.module_name = self.net.__class__.__name__

        self.is_warmup = is_warmup

        # loss function
        self.metric = metric
        self.best_result = MinMetric()
        self.save_dir = save_dir

        if is_master():
            self.fieldnames = ["epoch"] + list(self.metric.metrics.keys())
            self.val_csv = os.path.join(self.save_dir, "val.csv")
            self.figure_dir = os.path.join(self.save_dir, "val_results")
            os.makedirs(self.figure_dir, exist_ok=True)

    def forward(self, batch):
        return self.net(batch)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        if self.is_warmup:
            self.warmup_iters = len(self.trainer.train_dataloader)
        if is_master():
            with open(self.val_csv, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
        self.avg_loss = 0
        self.count = 0

        self.best_result.reset()

    def training_step(self, batch: Any, batch_idx: int):
        _, loss, loss_val = self.forward(batch)
        self.avg_loss = (self.avg_loss * self.count + loss.item()) / (self.count + 1)
        self.count += 1
        self.log("avg_loss", self.avg_loss, on_epoch=False, prog_bar=True)
        self.log_dict(loss_val, on_epoch=False, prog_bar=False)
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        self.metric.reset()

    def on_validation_epoch_start(self):
        self.output_csv = os.path.join(self.save_dir, "output.csv")
        with open(self.output_csv, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def validation_step(self, batch: Any, batch_idx: int):
        gt = batch["gt"]
        pred, _, loss_val = self.forward(batch)
        self.metric.evaluate(pred, gt)
        rank, word_size = get_dist_info()
        result = {}
        result["epoch"] = batch_idx * word_size + rank
        for k in self.metric.metrics.keys():
            result[k] = f"{self.metric.metrics[k].item():.5f}"
        with open(self.output_csv, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(result)
        self.log_dict(
            loss_val, on_epoch=False, sync_dist=True, prog_bar=False, on_step=True
        )
        if batch_idx == 0:
            if is_master():
                out_path = os.path.join(
                    self.figure_dir, f"epoch_{self.current_epoch}.png"
                )
                if self.hparams.dataset in ["nyu", "sunrgbd"]:
                    batch_save(
                        batch["rgb"],
                        batch["dep"],
                        pred,
                        gt,
                        (pred - gt).abs(),
                        out_path,
                        self.hparams.dataset,
                    )
                elif self.hparams.dataset == "kitti":
                    batch_save_kitti(
                        batch["rgb"],
                        batch["dep"],
                        pred,
                        gt,
                        (pred - gt).abs(),
                        out_path,
                    )

    def validation_epoch_end(self, outputs: List[Any]):
        avg_metric = self.metric.average()
        for key in avg_metric.keys():
            avg_metric[key] = reduce_value(avg_metric[key])
            self.log(f"val/{key}", avg_metric[key], on_step=False, on_epoch=True)
        self.best_result(avg_metric[self.hparams.monitor])
        self.log(
            "val/best_result", self.best_result.compute(), prog_bar=True, on_epoch=True
        )
        self.metric.reset()
        avg_metric["epoch"] = self.current_epoch
        self.save_val_result(avg_metric)
        try:
            self.lr = self.optimizers().param_groups[0]["lr"]
        except:
            pass

    def on_test_start(self):
        super().on_test_start()
        self.test_out_dir = os.path.join(self.save_dir, "test")
        os.makedirs(self.test_out_dir, exist_ok=True)
        self.test_csv = os.path.join(self.test_out_dir, f"{self.module_name}.csv")
        self.test_fieldnames = ["filename"] + list(self.metric.metrics.keys())
        with open(self.test_csv, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.test_fieldnames)
            writer.writeheader()

    def test_step(self, batch: Any, batch_idx: int):
        rank, word_size = get_dist_info()
        pred, _, loss_val = self.forward(batch)
        self.log_dict(
            loss_val, on_epoch=False, sync_dist=True, prog_bar=False, on_step=True
        )
        if self.hparams.dataset in ["nyu", "sunrgbd"]:
            gt = batch["gt"]
            self.metric.evaluate(pred, gt)
            result = {}
            result["filename"] = batch_idx * word_size + rank
            for k in self.metric.metrics.keys():
                result[k] = f"{self.metric.metrics[k].item():.5f}"
            with open(self.test_csv, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.test_fieldnames)
                writer.writerow(result)
        str_i = str(batch_idx * word_size + rank)
        path_i = str_i.zfill(10) + ".png"
        path = os.path.join(self.test_out_dir, path_i)
        if self.hparams.dataset == "kitti":
            save_depth_as_uint16png_upload(pred, path)
        else:
            image = merge_into_row(
                batch["rgb"],
                batch["dep"],
                pred,
                gt,
                (pred - gt).abs(),
                self.hparams.dataset,
            )
            save_image(image, path)

    def test_epoch_end(self, outputs: List[Any]):
        if self.hparams.dataset in ["nyu", "sunrgbd"]:
            avg_metric = self.metric.average()
            for key in avg_metric.keys():
                avg_metric[key] = reduce_value(avg_metric[key])
                self.log(
                    f"test/{key}",
                    avg_metric[key],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            self.metric.reset()
            avg_metric["epoch"] = "Test"
            self.save_val_result(avg_metric)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
        optimizer = self.hparams.optimizer(params)
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val/{self.hparams.monitor}",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # warm_up
        if self.is_warmup:
            if self.trainer.global_step <= self.warmup_iters:
                lr_warm_up = (
                    self.hparams.base_lr * self.trainer.global_step / self.warmup_iters
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_warm_up

    def save_val_result(self, result):
        if is_master():
            with open(self.val_csv, "a") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                for key in result.keys():
                    if key != "epoch":
                        result[key] = "{:.5f}".format(result[key].data)
                writer.writerow(result)


if __name__ == "__main__":
    _ = DepthLitModule(None, None, None)
