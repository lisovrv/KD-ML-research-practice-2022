import os
import shutil
from argparse import ArgumentParser
from inspect import getfullargspec

import torch
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


class Experiment(LightningModule):
    """
    Main class. All experiments should inherit from it.
    """

    @classmethod
    def get_parser(cls):
        # special thanks to https://linuxtut.com/en/1b2e76f3bfd18dcc1975/
        args, _, _, defaults, _, _, _, = getfullargspec(cls.__init__)
        args = args[1:]  # skip self
        assert len(args) == len(defaults)

        parser = ArgumentParser()
        for arg, default in zip(args, defaults):
            if type(default) is not bool:
                parser.add_argument("--" + arg.replace("_", "-"), type=type(default), default=default)
                continue

            # special case for boolean arguments
            if default is True:
                parser.add_argument("--disable-" + arg.replace("_", "-"), dest=arg, default=True, action="store_false")
            else:
                parser.add_argument("--enable-" + arg.replace("_", "-"), dest=arg, default=False, action="store_true")

        return parser

    @classmethod
    def main(cls):
        parser = cls.get_parser()
        args = parser.parse_args()
        experiment = cls(**vars(args))
        experiment.run()

    def run(self, log_every_n_steps: int = 5):
        if hasattr(self, "artifacts_path") and os.path.exists(self.artifacts_path):
            shutil.rmtree(self.artifacts_path)
            os.mkdir(self.artifacts_path)

        logger = WandbLogger(name=self.hparams.experiment_name, project=self.hparams.wandb_project)
        trainer = Trainer(
            logger,
            gpus=int(torch.cuda.is_available()),
            auto_lr_find=True,
            max_epochs=self.hparams.epochs,
            log_every_n_steps=log_every_n_steps,
        )
        trainer.fit(self)
        wandb.finish()

        if hasattr(self, "artifacts_path"):
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
