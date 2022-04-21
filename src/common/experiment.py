import os
import shutil
from argparse import ArgumentParser
from inspect import getfullargspec
from typing import Any, Type

import torch
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from .typing_utils import get_args, is_optional


class ExperimentArgumentParser(ArgumentParser):
    def __init__(self, experiment_class: Type["Experiment"]):
        super().__init__()
        # special thanks to https://linuxtut.com/en/1b2e76f3bfd18dcc1975/
        argspec = getfullargspec(experiment_class.__init__)
        args = argspec.args[1:]  # skip self
        defaults = argspec.defaults
        annotations = argspec.annotations

        assert len(args) == len(defaults)

        arg_types = {}  # maps argument name to its type
        arg_defaults = {}  # maps argument name to its default value
        for arg, default in zip(args, defaults):
            arg_defaults[arg] = default

            # infer argument type from annotations or default value
            arg_type = type(default)
            if arg in annotations:
                arg_type = annotations[arg]

                # special case of Optional arguments
                if is_optional(arg_type):
                    arg_type = get_args(arg_type)[0]

            arg_types[arg] = arg_type

        for arg in args:
            if arg_types[arg] is bool:
                self.add_bool_argument(arg, arg_defaults[arg])
            else:
                self.add_other_argument(arg, arg_types[arg], arg_defaults[arg])

    def add_bool_argument(self, arg: str, default: bool) -> None:
        if default is True:
            self.add_argument("--disable_" + arg, dest=arg, default=True, action="store_false")
        else:
            self.add_argument("--enable_" + arg, dest=arg, default=False, action="store_true")

    def add_other_argument(self, arg: str, type_: Type, default: Any) -> None:
        self.add_argument("--" + arg, type=type_, default=default)


class Experiment(LightningModule):
    """
    Main class. All experiments should inherit from it.
    """

    def __init__(self):
        super().__init__()
        self._train_data = None
        self._train_dataloader = None

        self._val_data = None
        self._val_dataloader = None

    @property
    def experiment_name(self):
        return self.hparams.get("experiment_name", "experiment")

    @property
    def wandb_project(self):
        return self.hparams.get("wandb_project", "project")

    @property
    def dataset(self):
        return self.hparams.get("dataset", "cifar100")

    @property
    def batch_size(self):
        return self.hparams.get("batch_size", 1024)

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

    @property
    def in_channels(self):
        if self.dataset == "cifar100":
            return 3
        elif self.dataset == "fashion_mnist":
            return 1
        else:
            raise ValueError(f"unsupported dataset {self.dataset}")

    @property
    def num_classes(self):
        if self.dataset == "cifar100":
            return 100
        elif self.dataset == "fashion_mnist":
            return 10
        else:
            raise ValueError(f"unsupported dataset {self.dataset}")

    @property
    def train_data(self):
        if self._train_data is None:
            if self.dataset == "cifar100":
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )
                self._train_data = CIFAR100(
                    root="data",
                    train=True,
                    transform=transform,
                    download=True,
                )
            elif self.dataset == "fashion_mnist":
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize((0.5,), (0.5,)),
                    ]
                )
                self._train_data = FashionMNIST(
                    root="data",
                    train=True,
                    transform=transform,
                    download=True,
                )
            else:
                raise ValueError(f"unsupported dataset {self.dataset}")

        return self._train_data

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(
                self.train_data,
                self.batch_size,
                shuffle=True,
            )

        return self._train_dataloader

    @property
    def val_data(self):
        if self._val_data is None:
            if self.dataset == "cifar100":
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )
                self._val_data = CIFAR100(
                    root="data",
                    train=False,
                    transform=transform,
                    download=True,
                )
            elif self.dataset == "fashion_mnist":
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize((0.5,), (0.5,)),
                    ]
                )
                self._val_data = FashionMNIST(
                    root="data",
                    train=False,
                    transform=transform,
                    download=True,
                )
            else:
                raise ValueError(f"unsupported dataset {self.dataset}")

        return self._val_data

    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(
                self.val_data,
                self.batch_size,
                shuffle=False,
            )

        return self._val_dataloader

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

    @property
    def artifacts_path(self):
        artifacts_base = self.hparams.get("artifacts_base", "./artifacts")
        return os.path.join(artifacts_base, self.experiment_name)

    @classmethod
    def main(cls):
        parser = ExperimentArgumentParser(cls)
        args = parser.parse_args()
        experiment = cls(**vars(args))
        experiment.run()

    def run(self, log_every_n_steps: int = 5):
        if os.path.exists(self.artifacts_path):
            shutil.rmtree(self.artifacts_path)
            os.mkdir(self.artifacts_path)

        logger = WandbLogger(name=self.experiment_name, project=self.wandb_project)
        checkpoint_callback = ModelCheckpoint(
            self.artifacts_path,
            "{epoch}-{train_acc:.2f}-{val_acc:.2f}",
            monitor="val_acc",
            save_weights_only=True,
            mode="max",
            save_top_k=10,
        )
        trainer = Trainer(
            logger,
            gpus=int(torch.cuda.is_available()),
            auto_lr_find=True,
            max_epochs=self.epochs,
            callbacks=[checkpoint_callback],
            log_every_n_steps=log_every_n_steps,
        )
        trainer.fit(self)
        wandb.finish()

        # At this point we have a directory full of lightning module checkpoints.
        # The problem with those is the fact that they tend to have lots of auxiliary stuff including teacher
        # model's state (for the distillated models). Most of the time it is a good idea to store all of it, e.g.
        # to fine tune model from the existing state, however, when it comes to using such models in other models
        # (e.g. as a teacher) it can become very tricky as loading the checkpoint requires the model to define
        # its teacher the same way it was defined during training. It becomes especially tricky when we want to
        # use one distilled model as a teacher for a new one (and use that one to train yet another, etc.).
        # On the other hand, after the training is done, we mostly care for the weights of the target (i.e. student)
        # model. So here I decouple those: I load every checkpoint and only re-save the .model or .student_model.

        model_attr = None  # only save weights of the trainable model
        if hasattr(self, "model"):
            model_attr = "model"
        if hasattr(self, "student_model"):
            model_attr = "student_model"
        if hasattr(self, "model_attr"):
            model_attr = self.model_attr

        if model_attr is None:
            raise ValueError(f"{type(self).__name__} has none of .model, .student_model and .model_attr attributes")

        for ckpt in os.scandir(self.artifacts_path):
            model_state = type(self).load_from_checkpoint(ckpt.path)
            model = getattr(model_state, model_attr)
            torch.save(model.state_dict(), ckpt.path)
