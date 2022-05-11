import os
import sys
from typing import Optional

import torch

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, resnet18  # noqa: E402


class TOPKLoss:
    """
    Teacher Aware Label Smoothing loss.

    The idea here is to interpret the knowledge distillation as a label smoothing technique.
    Namely, we can try and use label smoothing with the target probability given by the teacher model.
    """

    def __init__(self, k: int = 10, temperature: float = 4.0):
        self.k = k
        self.temperature = temperature

    def __call__(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        batch_size, n_classes = teacher_output.size()
        teacher_probs = torch.softmax(teacher_output / self.temperature, dim=1)

        teacher_argsort = torch.argsort(teacher_probs, dim=1)
        unsqueezed_range = torch.arange(batch_size)[:, None]
        topk_indices = teacher_argsort[:, -self.k :]  # noqa: E203
        topk_sum_probs = teacher_probs[unsqueezed_range, topk_indices].sum(dim=1, keepdim=True)

        target_probs = (1 - topk_sum_probs) / (n_classes - self.k)
        target_probs = target_probs.repeat(1, n_classes)
        target_probs[unsqueezed_range, topk_indices] = teacher_probs[unsqueezed_range, topk_indices]

        student_logprobs = torch.log_softmax(student_output / self.temperature, dim=1)
        return -(target_probs * student_logprobs).sum(dim=1).mean()


class TOPKDistillExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "topk_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        log_every_n_steps: int = 5,
        teacher_checkpoint: Optional[str] = None,
        k: int = 10,
        temperature: float = 4.0,
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
        self.criterion = TOPKLoss(k, temperature)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_output = self.teacher_model(image)

        loss = self.criterion(student_output, teacher_output)
        self.log("loss", loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_acc(student_output, target)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_output = self.teacher_model(image)

        loss = self.criterion(student_output, teacher_output)
        self.log("val_loss", loss)

        self.val_acc(student_output, target)
        self.log("val_acc", self.val_acc)


if __name__ == "__main__":
    TOPKDistillExperiment.main()
