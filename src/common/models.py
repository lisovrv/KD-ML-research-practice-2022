from typing import Optional

import timm
import torch


def create_model(model_name: str, num_classes: int = 100, checkpoint_path: Optional[str] = None, freeze: bool = False):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

    return model


def resnet18(num_classes: int = 100, checkpoint_path: Optional[str] = None, freeze: bool = False):
    return create_model("resnet18", num_classes, checkpoint_path, freeze)
