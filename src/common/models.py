from typing import Optional

import timm
import torch
import torch.nn as nn


class PretrainedModel(nn.Module):
    def __init__(self, model: nn.Module, checkpoint_path: str):
        super().__init__()
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.frozen = False
        self.freeze()

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.frozen = True

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

        self.model.train()
        self.frozen = False

    def forward(self, *args, **kwargs):
        # Do not forward in training mode if model is frozen.
        if self.frozen and self.model.training:
            self.model.eval()

        return self.model(*args, **kwargs)


def create_model(model_name: str, num_classes: int = 100, checkpoint_path: Optional[str] = None):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if checkpoint_path is not None:
        model = PretrainedModel(model, checkpoint_path)

    return model


def resnet18(num_classes: int = 100, checkpoint_path: Optional[str] = None):
    return create_model("resnet18", num_classes, checkpoint_path)


def resnet50(num_classes: int = 100, checkpoint_path: Optional[str] = None):
    return create_model("resnet50", num_classes, checkpoint_path)
