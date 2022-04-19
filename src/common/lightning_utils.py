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


class DataMixin(LightningModule):
    def __init__(self):
        super().__init__()
        self._train_dataloader = None
        self._val_dataloader = None

    @property
    def dataset(self):
        return self.hparams.get("dataset", "cifar100")

    @property
    def batch_size(self):
        return self.hparams.get("batch_size", 1024)

    def train_dataloader(self):
        if self._train_dataloader is None:
            if self.dataset == "cifar100":
                train_data = CIFAR100(
                    root="data",
                    train=True,
                    transform=ToTensor(),
                    download=True,
                )
            else:
                raise ValueError(f"unsupported dataset {self.dataset}")

            self._train_dataloader = DataLoader(
                train_data,
                self.batch_size,
                shuffle=True,
            )

        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            if self.dataset == "cifar100":
                val_data = CIFAR100(
                    root="data",
                    train=False,
                    transform=ToTensor(),
                    download=True,
                )
            else:
                raise ValueError(f"unsupported dataset {self.dataset}")

            self._val_dataloader = DataLoader(
                val_data,
                self.batch_size,
                shuffle=False,
            )

        return self._val_dataloader


class OptimizerMixin(LightningModule):
    @property
    def lr(self):
        return self.hparams.get("lr", 0.001)

    @property
    def weight_decay(self):
        return self.hparams.get("weight_decay", 0.01)

    @property
    def epochs(self):
        return self.hparams.get("epochs", 50)

    @property
    def optimizer(self):
        return self.hparams.get("optimizer", "adam")

    @property
    def lr_scheduler(self):
        return self.hparams.get("lr_scheduler", None)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"unsupported optimizer {self.optimizer}")

        if self.lr_scheduler is None:
            return optimizer
        elif self.lr_scheduler == "one_cycle_lr":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.lr,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=self.epochs,
            )
            lr_scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        else:
            raise ValueError(f"unsupported lr scheduler {self.lr_scheduler}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ModelCheckpointMixin(LightningModule):
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
        artifacts_base = self.hparams.get("artifacts_base", "./artifacts")
        return os.path.join(artifacts_base, self.experiment_name)
