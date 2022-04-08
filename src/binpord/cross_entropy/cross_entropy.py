import torch.nn.functional as F

# For imports from common
# TODO: is there a better way?
import os, sys; sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))  # noqa
from common import Experiment, CIFAR100_Mixin, AdamWOneCycleLRMixin, ModelCheckpointMixin, Accuracy, resnet18


class CrossEntropyExperiment(Experiment, CIFAR100_Mixin, AdamWOneCycleLRMixin, ModelCheckpointMixin):
    def __init__(
            self,
            experiment_name: str = "cross_entropy",
            wandb_project: str = "kd-cifar100-resnet18",
            batch_size: int = 1024,
            epochs: int = 50,
            lr: float = 1e-3,
            weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)

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
