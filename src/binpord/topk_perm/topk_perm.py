import os
import sys
from typing import Optional

import torch
from torchdistill.losses.single import KDLoss

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, resnet18  # noqa: E402


class TopKPermLoss:
    def __init__(self, k: int = 10, temperature: float = 4.0):
        self.k = k
        self.kd_loss = KDLoss(temperature, alpha=0)

    def __call__(self, student_output: torch.Tensor, teacher_output: torch.Tensor):
        batch_size, n_classes = student_output.size()
        permutation = torch.stack(
            [torch.randperm(n_classes, device=student_output.device) for _ in range(batch_size)],
            dim=0,
        )

        # Here, permutation may permute one of top-k classes. I need to fix that.
        self._fix_permutation(permutation, teacher_output)

        teacher_output = self._apply_permutation(teacher_output, permutation)
        return self.kd_loss(student_output, teacher_output)

    def _fix_permutation(self, permutation: torch.Tensor, teacher_output: torch.Tensor):
        teacher_argsort = torch.argsort(teacher_output, dim=1)
        topk_indices = teacher_argsort[:, -self.k :]  # noqa: E203
        for i in range(self.k):
            self._fix_indices(permutation, topk_indices[:, i])

    @staticmethod
    def _fix_indices(permutation: torch.Tensor, targets: torch.Tensor):
        batch_size = targets.size(0)
        _, indices = (permutation == targets[:, None]).nonzero(as_tuple=True)
        permuted_targets_indices = permutation[range(batch_size), targets]
        permutation[range(batch_size), targets] = targets
        permutation[range(batch_size), indices] = permuted_targets_indices

    @staticmethod
    def _apply_permutation(teacher_output: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return torch.gather(teacher_output, dim=1, index=permutation)


class TopKPermExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "dkpp_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        log_every_n_steps: int = 20,
        temperature: float = 4.0,
        k: int = 10,
        teacher_checkpoint: Optional[str] = None,
        batch_size: int = 256,
        epochs: int = 30,
        optimizer: str = "adam",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.015,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_model = resnet18()
        self.teacher_model = resnet18(checkpoint_path=teacher_checkpoint)
        self.criterion = TopKPermLoss(k, temperature)

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
    TopKPermExperiment.main()
