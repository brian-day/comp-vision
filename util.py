from enum import Enum
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate  # type: ignore
from torchmetrics import Accuracy, Precision, Recall

import matplotlib.pyplot as plt

from typing import Protocol, Tuple, List, Dict, Callable, Any, Optional, Literal

import logging


type Tensor_or_Array = torch.Tensor | np.ndarray


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


type LogLevelType = Literal[
    LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL
]
type StatsDict = Dict[str, float]


# NOTE: Variable names must be exact for any functions matching StatsFunc
class StatsFunc(Protocol):
    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        num_classes: int,
        *,
        device: Optional[torch.device] = None,
    ) -> StatsDict: ...


def setup_logger(log_file: str = "log.txt", level: LogLevelType = LogLevel.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level.value)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level.value)

    logger.addHandler(file_handler)

    return logger


def set_dtype_and_device_single(x: Tensor_or_Array, dtype=torch.float32, device="cpu"):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype).to(device)
    else:
        return x.to(dtype).to(device)


def set_dtype_and_device(
    x_train: Tensor_or_Array,
    x_test: Tensor_or_Array,
    y_train: Tensor_or_Array,
    y_test: Tensor_or_Array,
    dtype=torch.float32,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts input tensors to torch.float32 and moves them to the specified device.
    """
    x_train = set_dtype_and_device_single(x_train, dtype=dtype, device=device)
    x_test = set_dtype_and_device_single(x_test, dtype=dtype, device=device)
    y_train = set_dtype_and_device_single(y_train, dtype=dtype, device=device)
    y_test = set_dtype_and_device_single(y_test, dtype=dtype, device=device)

    return x_train, x_test, y_train, y_test


def train_model(
    model: nn.Module,
    train_dataset: Any,
    *,
    loss_fxn: torch.nn.Module = nn.BCEWithLogitsLoss(),
    optim_type=torch.optim.SGD,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    print_iter: Optional[int] = None,
    print_stats: bool = False,
    stats_fxn: Optional[StatsFunc] = None,
    print_test: bool = False,
    test_dataset: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[StatsDict]]:
    """Functional approach for training a pytorch model. Model parameters are updated in place.
    Currently expects that all inputs are already on the appropriate device.

    Args:
        model: Instance of model to be trained
        train_dataset: Dataset of the training data
        loss_fxn: Loss function used in optimizing model parameters. Defaults to nn.BCEWithLogitsLoss().
        optim_type: Optimizer type for updating model parameters. Defaults to torch.optim.SGD.
        learning_rate: Controls how aggressively optimizer updates model parameters during training. Defaults to 0.01.
        epochs: Number of iterations over the training dataset. Defaults to 100.
        batch_size: Number of samples per pass through training loop in each epoch. Number of passes per epoch is
            determined by the number of samples / batch size.
        print_iter: Controls how frequently model statistics are printed during testing. If None, no stats printed.
            Defaults to None.
        print_stats: Controls if results from stats_fxn are included in printed outputs. Defaults to False.
        stats_fxn: Function of additional statistics to be calculated during training. Has well defined input signature,
            and currently requires the output to be a dictionary with singleton numeric values. Defaults to None.
        print_test: Controls whether test data is evaluated and results printed during training. Evaluation of the test
            data only occurs on print iterations. If True, requires a test_dataset to be provided. Defaults to False.
        test_dataset: Dataset of the test data. Only required if print_test is True.

    Returns:
        Returns a tuple containing the loss score and the other statistics at each iteration (epoch, not batch).
    """
    logger = logging.getLogger(__name__)  # Check type and root logger.

    # Input validation in manual function replaced by proper type annotations :)

    # Convert dataset to a data loader with appropriate batch size.
    # Always shuffles training data, never shuffles tests data.
    model.to(device)
    num_classes = len(train_dataset.classes)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: (batch_.to(device) for batch_ in default_collate(batch)),
    )

    # Setup Optimizer
    optim = optim_type(params=model.parameters(), lr=learning_rate)

    # Train the Model
    loss_over_time: List[float] = []
    stats_over_time: list[StatsDict] = []
    train_stats: StatsDict = {}
    for epoch in range(epochs):
        model.train()
        train_loss = torch.Tensor([0])
        for batch, (x, y) in enumerate(train_data_loader):
            y_pred = model(x)
            loss = loss_fxn(y_pred, y)
            train_loss += loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            if stats_fxn:
                if batch == 0:
                    train_stats = stats_fxn(y_pred, y, num_classes, device=device)
                else:
                    curr_stats = stats_fxn(y_pred, y, num_classes, device=device)
                    for key in train_stats:
                        train_stats[key] += curr_stats[key]

        # Calculate epoch stats
        train_loss /= len(train_data_loader)
        loss_over_time.append(train_loss.item())
        if stats_fxn:
            for key in train_stats:
                train_stats[key] /= len(train_data_loader)
                stats_over_time.append(train_stats)

        # Print results, evaluating test data if necessary
        if print_iter and epoch % print_iter == 0:
            print_str = f"Epoch: {epoch} | train_loss: {train_loss.item():.4f}"
            if print_stats and stats_fxn:
                stats_print_str = generate_stats_print_string(train_stats)
                print_str += f", stats: {stats_print_str}"
            if print_test and test_dataset:
                # Using separate calls in case stats function is expensive.
                if print_stats and stats_fxn:
                    # TODO: Update stats calculations in batch loop to be more flexible for non-singleton stats.
                    test_loss, test_stats = eval_model(
                        model,
                        test_dataset,
                        loss_fxn=loss_fxn,
                        batch_size=batch_size,
                        stats_fxn=stats_fxn,
                    )
                    stats_print_str = generate_stats_print_string(test_stats)
                    print_str += f" | test_loss: {test_loss:.4f}, test_stats: {stats_print_str}"
                else:
                    test_loss, _ = eval_model(
                        model, test_dataset, loss_fxn=loss_fxn, batch_size=batch_size
                    )
                    print_str += f" | test_loss: {test_loss:.4f}"

            print(print_str)
            logger.info(print_str)

    return loss_over_time, stats_over_time


def eval_model(
    model: nn.Module,
    test_dataset: Any,
    *,
    loss_fxn: torch.nn.Module = nn.BCEWithLogitsLoss(),
    batch_size: int = 32,
    stats_fxn: Optional[StatsFunc] = None,
) -> Tuple[float, StatsDict]:
    num_classes = len(test_dataset.classes)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    test_stats: StatsDict = {}
    with torch.inference_mode():
        test_loss = torch.tensor([0])
        for batch, (x, y) in enumerate(test_data_loader):
            y_pred = model(x)
            test_loss += loss_fxn(y_pred, y)

            # FIXME: Feels awkward, and not flexible for non-singleton stats.
            if stats_fxn:
                if batch == 0:
                    test_stats = stats_fxn(y_pred, y, num_classes)
                else:
                    curr_stats = stats_fxn(y_pred, y, num_classes)
                    for key in test_stats:
                        test_stats[key] += curr_stats[key]

        test_loss /= len(test_data_loader)
        for key in test_stats:
            test_stats[key] /= len(test_data_loader)

    return test_loss.item(), test_stats


def generate_stats_print_string(stats_dict: StatsDict) -> str:
    print_items: List[str] = []
    for key, val in stats_dict.items():
        print_items.append(f"{key}: {val:.4f}")

    return ", ".join(print_items)


def multiclass_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    *,
    device: Optional[torch.device] = None,
) -> StatsDict:
    # See: https://github.com/Lightning-AI/torchmetrics/issues/2280
    # sklearn has alternative metrics library
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    pre = Precision(task="multiclass", average="macro", num_classes=num_classes)
    rec = Recall(task="multiclass", average="macro", num_classes=num_classes)

    if device:
        acc.to(device)
        pre.to(device)
        rec.to(device)

    acc_val = acc(y_pred, y_true).item()
    pre_val = pre(y_pred, y_true).item()
    rec_val = rec(y_pred, y_true).item()

    return {"Accuracy": acc_val, "Precision": pre_val, "Recall": rec_val}


# Generic Utility Function
def walk_through_dir(dir_path: Path) -> None:
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str or pathlib.Path): target directory

    Returns:
        A print out of:
            number of subdiretories in dir_path
            number of images (files) in each subdirectory
            name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Generic utility function
def find_classes(directory: Path) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
