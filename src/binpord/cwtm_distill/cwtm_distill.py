import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, resnet18  # noqa: E402


class CWTMLoss:
    """
    Confidence Weighted by Teacher Max loss as described in the Born Again Networks paper (arXiv:1805.04770).

    The idea here is to weight the student cross-entropy loss wrt the confidence of teacher in the prediction.
    Here the confidence is measured via the maximum of teacher probabilities.
    The maximum operation is a suspicious one, so along with this experiment I also provide results of the
    similar experiment Confidence Weighted by Teacher Confidence (CWTC) where I weight the examples based on
    teacher confidence of the true label.
    """

    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def __call__(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ):
        element_wise_loss = self.cross_entropy(student_output, target)
        teacher_probabilities = torch.softmax(teacher_output, dim=1)
        teacher_max_confidences = teacher_probabilities.max(dim=1)[0]
        weights = teacher_max_confidences / teacher_max_confidences.sum()
        return (weights * element_wise_loss).sum()


class CWTMDistillExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "cwtm_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        log_every_n_steps: int = 5,
        teacher_checkpoint: Optional[str] = None,
        batch_size: int = 1024,
        epochs: int = 50,
        optimizer: str = "adamw",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.001,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_model = resnet18()
        self.teacher_model = resnet18(checkpoint_path=teacher_checkpoint)
        self.criterion = CWTMLoss()

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
    CWTMDistillExperiment.main()
