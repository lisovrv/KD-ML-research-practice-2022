import multiprocessing
import typing as tp
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision
from tqdm import tqdm, trange
from typing_extensions import Protocol

pd.set_option('max_columns', None)
REPO_NAME = "chenyaofo/pytorch-cifar-models"
TEACHER_MODEL = "cifar100_resnet56"
STUDENT_MODEL = "cifar100_resnet20"
BATCH_SIZE = 2048
NUM_WORKERS = multiprocessing.cpu_count() - 1
NUM_EPOCHS = 500
SCHEDULER_STEP = 100
LOGS_DIR = Path("./cifar_logs/")
ALPHA = 0.1
T = 10

LOGS_DIR.mkdir(parents=True, exist_ok=True)


class LossCallable(Protocol):

    def __init__(self, *args, **kwargs) -> None:
        ...

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        ...


class ModelsProvider:

    def __init__(self, repo_name: str):
        self._repo_name = repo_name

    def get_all_available_models(self) -> tp.List[str]:
        return torch.hub.list(self._repo_name, force_reload=True)

    def get_model_by_name(self, model_name: str, pretrained: bool = True) -> nn.Module:
        model: nn.Module = torch.hub.load(self._repo_name, model_name, pretrained=pretrained)
        return model


class Runner:

    def __init__(
        self, loader_train: data.DataLoader, loader_test: data.DataLoader, device: torch.device = torch.device('cpu')
    ) -> None:
        self._loader_train = loader_train
        self._loader_test = loader_test
        self._device = device

    def validate_model(self, model: nn.Module, verbose: bool = False) -> float:
        model.eval()
        model.train(False)

        total_accuracy = 0
        total_count = 0

        data_iterator = self._loader_test
        if verbose:
            data_iterator = tqdm(data_iterator)

        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                input = batch_x.to(self._device)
                target = batch_y.to(self._device)

                output: torch.Tensor = model(input)
                output_argmax_indices = torch.argmax(output, dim=1)
                accuracy = (output_argmax_indices == target).sum().item()
                total_accuracy += accuracy
                total_count += input.shape[0]

        return total_accuracy / total_count

    def train_model(
        self,
        model: nn.Module,
        loss_function: LossCallable,
        optimizer: optim.Optimizer,
        oclr: tp.Optional[lr_scheduler.OneCycleLR] = None,
        verbose: bool = False
    ) -> tp.Tuple[float, float]:
        model.train(True)

        total_accuracy = 0
        total_loss = 0.0
        total_count = 0

        data_iterator = self._loader_train
        if verbose:
            data_iterator = tqdm(data_iterator)

        for batch_x, batch_y in data_iterator:
            optimizer.zero_grad()
            input = batch_x.to(self._device)
            target = batch_y.to(self._device)

            output: torch.Tensor = model(input)
            loss = loss_function(prediction=output, target=target, input=input)
            loss.backward()
            optimizer.step()
            if oclr is not None:
                oclr.step()

            output_argmax_indices = torch.argmax(output, dim=1)
            accuracy = (output_argmax_indices == target).sum().item()

            total_loss += loss.item()
            total_accuracy += accuracy
            total_count += input.shape[0]

        return total_loss / total_count, total_accuracy / total_count


class CrossEntropyLoss(LossCallable):

    def __init__(self) -> None:
        self._loss = nn.CrossEntropyLoss()

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        return self._loss(prediction, target)


class DistillationLoss(LossCallable):

    def __init__(self, teacher: nn.Module, alpha: float = 0.2, T: float = 5) -> None:
        self._teacher = teacher
        self._alpha = alpha
        self._T = T

        self._ce = nn.CrossEntropyLoss(reduction='sum')
        self._bce = nn.BCELoss(reduction='sum')

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        prediction_softmax = F.softmax(prediction / self._T, dim=1)
        with torch.no_grad():
            teacher_scores = self._teacher(input)
        teacher_scores_softmax = F.softmax(teacher_scores / self._T, dim=1)

        soft_loss = self._bce(prediction_softmax, teacher_scores_softmax)
        hard_loss = self._ce(prediction, target)
        loss = self._alpha * hard_loss + (1 - self._alpha) * soft_loss
        return loss


if __name__ == "__main__":
    start_time = datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provider = ModelsProvider(REPO_NAME)
    transform = torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = torchvision.datasets.CIFAR100(root="/tmp", train=True, transform=transform, download=True)
    dataset_test = torchvision.datasets.CIFAR100(root="/tmp", train=False, transform=transform, download=True)

    loader_train = data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    loader_test = data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    teacher = provider.get_model_by_name(TEACHER_MODEL, pretrained=False).to(device=device)
    student_pretrained = provider.get_model_by_name(STUDENT_MODEL).to(device=device)
    teacher.eval()

    student_for_distillation = provider.get_model_by_name(STUDENT_MODEL, pretrained=False).to(device=device)
    student_for_training = provider.get_model_by_name(STUDENT_MODEL, pretrained=False).to(device=device)

    runner = Runner(loader_train=loader_train, loader_test=loader_test, device=device)

    score = runner.validate_model(teacher, verbose=True)
    print(f"Teacher origin score: {score}")

    # ----------------------------------------------------------------------
    optimizer_distil = optim.AdamW(student_for_distillation.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer_train = optim.AdamW(student_for_training.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer_teacher = optim.AdamW(teacher.parameters(), lr=1e-4, weight_decay=0.01)
    # sheduler_distil = optim.lr_scheduler.StepLR(optimizer=optimizer_distil, step_size=SCHEDULER_STEP, gamma=0.3)
    # sheduler_train = optim.lr_scheduler.StepLR(optimizer=optimizer_train, step_size=SCHEDULER_STEP, gamma=0.3)
    scheduler_distil = lr_scheduler.OneCycleLR(
        optimizer=optimizer_distil, max_lr=1e-3, epochs=NUM_EPOCHS, steps_per_epoch=len(loader_train), final_div_factor=1000
    )
    scheduler_train = lr_scheduler.OneCycleLR(
        optimizer=optimizer_train, max_lr=1e-3, epochs=NUM_EPOCHS, steps_per_epoch=len(loader_train), final_div_factor=1000
    )
    scheduler_teacher  = lr_scheduler.OneCycleLR(
        optimizer=optimizer_teacher, max_lr=1e-3, epochs=NUM_EPOCHS, steps_per_epoch=len(loader_train), final_div_factor=1000
    )

    train_loss_function = CrossEntropyLoss()
    distillation_loss_function = DistillationLoss(teacher=teacher, alpha=ALPHA, T=T)

    results: tp.Dict[str, tp.List[float]] = {
        "epoch": [],
        "train_accuracy_simple": [],
        "train_accuracy_distil": [],
        "train_loss_simple": [],
        "train_loss_distil": [],
        "valid_accuracy_simple": [],
        "valid_accuracy_distil": []
    }

    for epoch in trange(NUM_EPOCHS):
        teacher_train_loss, teacher_train_accuracy = runner.train_model(
            teacher, loss_function=train_loss_function, optimizer=optimizer_teacher, oclr=scheduler_teacher
        )
        teacher_valid_accuracy = runner.validate_model(student_for_training)

        train_loss_simple, train_accuracy_simple = runner.train_model(
            student_for_training, loss_function=train_loss_function, optimizer=optimizer_train, oclr=scheduler_train
        )
        valid_accuracy_simple = runner.validate_model(student_for_training)

        train_loss_distil, train_accuracy_distil = runner.train_model(
            student_for_distillation, loss_function=distillation_loss_function, optimizer=optimizer_distil, oclr=scheduler_distil
        )
        valid_accuracy_distil = runner.validate_model(student_for_distillation)

        results['epoch'].append(epoch)

        results['teacher_train_loss'] = teacher_train_loss
        results['teacher_train_accuracy'] = teacher_train_accuracy
        results['teacher_valid_accuracy'] = teacher_valid_accuracy

        results['train_accuracy_simple'].append(train_accuracy_simple)
        results['train_loss_simple'].append(train_loss_simple)
        results['valid_accuracy_simple'].append(valid_accuracy_simple)

        results['train_accuracy_distil'].append(train_accuracy_distil)
        results['train_loss_distil'].append(train_loss_distil)
        results['valid_accuracy_distil'].append(valid_accuracy_distil)

        results_df = pd.DataFrame.from_dict(results)
        print(results_df)
        results_df.to_csv(str(LOGS_DIR / f"{start_time}_{TEACHER_MODEL}_{STUDENT_MODEL}_{ALPHA}_{T}.csv"))
