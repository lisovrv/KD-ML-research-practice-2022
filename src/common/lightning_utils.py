import os

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor


class Accuracy(Metric):
    """
    Computes accuracy from raw model logits.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        prediction = output.argmax(dim=1)
        self.correct += (prediction == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class CIFAR100_Mixin(LightningModule):  # noqa, _ is to distinguish number from Mixin
    def __init__(self):
        super().__init__()
        self._train_dataloader = None
        self._val_dataloader = None

    def train_dataloader(self):
        if self._train_dataloader is None:
            train_data = CIFAR100(
                root="data",
                train=True,
                transform=ToTensor(),
                download=True,
            )
            self._train_dataloader = DataLoader(
                train_data,
                self.hparams.batch_size,
                shuffle=True,
            )

        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            val_data = CIFAR100(
                root="data",
                train=False,
                transform=ToTensor(),
                download=True,
            )
            self._val_dataloader = DataLoader(
                val_data,
                self.hparams.batch_size,
                shuffle=False,
            )

        return self._val_dataloader


class AdamWOneCycleLRMixin(LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.lr,
            steps_per_epoch=len(self.train_dataloader()),
            epochs=self.hparams.epochs,
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ModelCheckpointMixin(LightningModule):
    def __init__(self):
        super().__init__()
        self._artifacts_path = None

    def configure_callbacks(self):
        return ModelCheckpoint(
            self.artifacts_path,
            "{epoch}-{train_acc:.2f}-{val_acc:.2f}",
            monitor="val_acc",
            save_weights_only=True,
            mode="max",
            save_top_k=10,
        )

    @property
    def artifacts_path(self):
        if self._artifacts_path is None:
            self._artifacts_path = os.path.join("./artifacts", self.hparams.experiment_name)

        return self._artifacts_path
