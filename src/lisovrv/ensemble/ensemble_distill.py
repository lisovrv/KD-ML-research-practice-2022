import os
import sys
from typing import Optional

from torch import nn
from torchdistill.losses.single import KDLoss

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import Accuracy, Experiment, create_model  # noqa: E402


class EnsembleDistillExperiment(Experiment):
    def __init__(
        self,
        experiment_name: str = "resnet18_x2-to-resnet18",
        wandb_project: str = "kd-cifar100-ensemble",
        temperature: float = 4.0,
        teacher_checkpoints: Optional[str] = None,
        n_teachers: int = 2,
        epochs: int = 50,
        optimizer: str = "adamw",
        lr_scheduler: str = "one_cycle_lr",
        lr: float = 0.015,
        weight_decay: float = 0.18,
        batch_size: int = 256,
        log_every_n_steps: int = 30,
        teacher_name: str = "resnet18",
        student_name: str = "resnet18",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_teachers = n_teachers

        self.student_model = create_model(student_name)

        teacher_checkpoints = teacher_checkpoints.split()
        assert len(teacher_checkpoints) == self.n_teachers
        self.teacher_models = nn.ModuleList(
            create_model(teacher_name, checkpoint_path=teacher_checkpoint) for teacher_checkpoint in teacher_checkpoints
        )

        self.criterion = KDLoss(temperature, alpha=0)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_outputs = [teacher_model(image) for teacher_model in self.teacher_models]

        loss = 0
        for teacher_output in teacher_outputs:
            loss += self.criterion(student_output, teacher_output)
        loss /= self.n_teachers

        self.log("loss", loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        self.train_acc(student_output, target)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        student_output = self.student_model(image)
        teacher_outputs = [teacher_model(image) for teacher_model in self.teacher_models]

        loss = 0
        for teacher_output in teacher_outputs:
            loss += self.criterion(student_output, teacher_output)
        loss /= self.n_teachers

        self.log("val_loss", loss)

        self.val_acc(student_output, target)
        self.log("val_acc", self.val_acc)


if __name__ == "__main__":
    EnsembleDistillExperiment.main()
