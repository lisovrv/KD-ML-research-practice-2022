import os
import sys
from typing import Optional

import torch

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, DataMixin, Experiment, ModelCheckpointMixin, OptimizerMixin, resnet18  # noqa: E402


class TALSLoss:
    """
    Teacher Aware Label Smoothing loss.

    The idea here is to interpret the knowledge distillation as a label smoothing technique.
    Namely, we can try and use label smoothing with the target probability given by the teacher model.
    """

    @staticmethod
    def __call__(student_output: torch.Tensor, teacher_output: torch.Tensor, targets: torch.Tensor):
        batch_size, n_classes = student_output.size()
        teacher_probs = torch.softmax(teacher_output, dim=1)
        teacher_probs = teacher_probs[range(batch_size), targets]
        smooth_probs = (1 - teacher_probs) / (n_classes - 1)

        target_probs = smooth_probs[:, None].repeat(1, n_classes)
        target_probs[range(batch_size), targets] = teacher_probs

        student_logprobs = torch.log_softmax(student_output, dim=1)
        return -(target_probs * student_logprobs).sum(dim=1).mean()


class TALSDistillExperiment(Experiment, DataMixin, OptimizerMixin, ModelCheckpointMixin):
    def __init__(
        self,
        experiment_name: str = "tals_distill",
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
        self.criterion = TALSLoss()

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
    TALSDistillExperiment.main()
