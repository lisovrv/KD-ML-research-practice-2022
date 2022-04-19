import os
import sys
from typing import Optional

from torchdistill.losses.single import KDLoss

# For imports from common
# TODO: is there a better way?
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))

from common import (  # noqa: E402
    Accuracy,
    AdamWOneCycleLRMixin,
    CIFAR100_Mixin,
    Experiment,
    ModelCheckpointMixin,
    resnet18,
)


class HintonDistillExperiment(Experiment, CIFAR100_Mixin, AdamWOneCycleLRMixin, ModelCheckpointMixin):
    def __init__(
        self,
        experiment_name: str = "hinton_distill",
        wandb_project: str = "kd-cifar100-resnet18",
        artifacts_base: str = "./artifacts",
        temperature: float = 4.0,
        teacher_checkpoint: Optional[str] = None,
        batch_size: int = 1024,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student_model = resnet18()
        self.teacher_model = resnet18(checkpoint_path=teacher_checkpoint, freeze=True)
        self.criterion = KDLoss(temperature, alpha=0)

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
    HintonDistillExperiment.main()