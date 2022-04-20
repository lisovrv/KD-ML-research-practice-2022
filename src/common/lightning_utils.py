import torch
from torchmetrics import Metric


class Accuracy(Metric):
    """
    Computes accuracy from raw model logits.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: torch.Tensor, target: torch.Tensor):
        prediction = output.argmax(dim=1)
        self.correct += (prediction == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
