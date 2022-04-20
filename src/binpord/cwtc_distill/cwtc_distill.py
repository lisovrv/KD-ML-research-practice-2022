import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, resnet18  # noqa: E402


class CWTCLoss:
    """
    Confidence Weighted by Teacher Confidence loss.

    The idea here is to weight the student cross-entropy loss wrt the confidence of teacher in the prediction.
    Here the confidence is measured via the probability of the target label given by teacher probabilities.
    This is accompanying experiment to the CWTM. More on motivation can be found there.
    """

    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def __call__(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ):
        batch_size = target.size(0)
        element_wise_loss = self.cross_entropy(student_output, target)

        teacher_probabilities = torch.softmax(teacher_output, dim=1)
        teacher_confidences = teacher_probabilities[range(batch_size), target]
        weights = teacher_confidences / teacher_confidences.sum()
        return (weights * element_wise_loss).sum()


class CWTCDistillExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "cwtc_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        teacher_checkpoint: Optional[str] = None,
        epochs: int = 50,
        optimizer: str = "adamw",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.001,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_model = resnet18()
        self.teacher_model = resnet18(checkpoint_path=teacher_checkpoint, freeze=True)
        self.criterion = CWTCLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_output = self.teacher_model(image)

        loss = self.criterion(student_output, teacher_output, target)
        self.log("loss", loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_acc(student_output, target)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_output = self.teacher_model(image)

        loss = self.criterion(student_output, teacher_output, target)
        self.log("val_loss", loss)

        self.val_acc(student_output, target)
        self.log("val_acc", self.val_acc)


if __name__ == "__main__":
    CWTCDistillExperiment.main()
