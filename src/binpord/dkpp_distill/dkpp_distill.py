import os
import sys
from typing import Optional

import torch
from torchdistill.losses.single import KDLoss

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, resnet18  # noqa: E402


class DKPPLoss:
    """
    Dark Knowledge with Permuted Predictions loss as described in Born Again Networks paper (arXiv:1805.04770).

    The idea here is to prove that the dark knowledge doesn't play any role in KD success via permuting the
    non-true classes predictions in a random fashion.
    Unfortunately the paper doesn't give many details on the subject, so I had to come up with specifics myself.
    As such this class has two modes: persistent and non-persistent permutations, and I experiment with both.
    """

    def __init__(self, temperature, persistent_permutation=True):
        self.kd_loss = KDLoss(temperature, alpha=0)
        self.persistent_permutation = persistent_permutation
        self.permutation = None

    def __call__(self, student_output: torch.Tensor, teacher_output: torch.Tensor, targets: torch.Tensor):
        batch_size, n_classes = student_output.size()
        if self.persistent_permutation:
            if self.permutation is None:
                self.permutation = torch.randperm(n_classes, device=student_output.device)

            permutation = self.permutation[None, :].repeat(batch_size, 1)
        else:
            permutation = torch.stack(
                [torch.randperm(student_output.size(1), device=student_output.device) for _ in range(batch_size)],
                dim=0,
            )

        # Here, permutation may permute the target class. I need to fix that.
        self._fix_permutation(permutation, targets)

        teacher_output = self._apply_permutation(teacher_output, permutation)
        return self.kd_loss(student_output, teacher_output)

    @staticmethod
    def _fix_permutation(permutation: torch.Tensor, targets: torch.Tensor):
        batch_size = targets.size(0)
        _, indices = (permutation == targets[:, None]).nonzero(as_tuple=True)
        permuted_targets_indices = permutation[range(batch_size), targets]
        permutation[range(batch_size), targets] = targets
        permutation[range(batch_size), indices] = permuted_targets_indices

    @staticmethod
    def _apply_permutation(teacher_output: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        return torch.gather(teacher_output, dim=1, index=permutation)


class DKPPDistillExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "dkpp_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        temperature: float = 4.0,
        persistent_permutation: bool = True,
        teacher_checkpoint: Optional[str] = None,
        dataset: str = "cifar100",
        epochs: int = 50,
        optimizer: str = "adamw",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.001,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_model = resnet18(self.num_classes, self.in_channels)
        self.teacher_model = resnet18(self.num_classes, self.in_channels, checkpoint_path=teacher_checkpoint)
        self.criterion = DKPPLoss(temperature, persistent_permutation)

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
    DKPPDistillExperiment.main()
