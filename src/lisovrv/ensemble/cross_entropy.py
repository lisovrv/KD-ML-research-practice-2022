import os
import sys

import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, create_model  # noqa: E402


class CrossEntropyExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "cross-entropy-training",
        wandb_project: str = "kd-cifar100-ensemble",
        epochs: int = 50,
        optimizer: str = "adamw",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.015,
        weight_decay: float = 0.18,
        batch_size: int = 256,
        log_every_n_steps: int = 30,
        model_name: str = "resnet18",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)

        loss = F.cross_entropy(output, target)
        self.log("loss", loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_acc(output, target)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)

        loss = F.cross_entropy(output, target)
        self.log("val_loss", loss)

        self.val_acc(output, target)
        self.log("val_acc", self.val_acc)


if __name__ == "__main__":
    CrossEntropyExperiment.main()
