from collections.abc import Iterator
from typing import Optional

import torch
from torch import nn

import bioplnn.utils.dataloaders as dataloaders


def initialize_dataloader(
    *, dataset: str, seed: Optional[int] = None, **kwargs
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Initialize a dataloader for a given dataset.

    Args:
        dataset (str): The dataset to use.
        seed (int, optional): The seed to use for the dataloader. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the dataloader.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            The train and validation dataloaders.
    """

    return getattr(dataloaders, f"get_{dataset}_dataloaders")(
        **kwargs, seed=seed
    )


def initialize_model(*, class_name: str, **kwargs) -> nn.Module:
    """Initialize a model based on the class name.

    Args:
        class_name (str): The name of the model class to use.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        nn.Module: The initialized model.
    """
    import bioplnn.models

    return getattr(bioplnn.models, class_name)(**kwargs)


def initialize_optimizer(
    *, class_name: str, model_parameters: Iterator[nn.Parameter], **kwargs
) -> torch.optim.Optimizer:
    """Initialize an optimizer for model training.

    Args:
        class_name (str): The name of the optimizer class to use.
        model_parameters (nn.ParameterList): The model parameters to optimize.
        **kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    return getattr(torch.optim, class_name)(model_parameters, **kwargs)


def initialize_scheduler(
    *, class_name: str, optimizer: torch.optim.Optimizer, **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """Initialize a learning rate scheduler.

    Args:
        class_name (str): The name of the scheduler class to use.
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        **kwargs: Additional keyword arguments to pass to the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The initialized scheduler.
    """
    return getattr(torch.optim.lr_scheduler, class_name)(optimizer, **kwargs)


def initialize_criterion(*, class_name: str, **kwargs) -> torch.nn.Module:
    """Initialize a loss criterion.

    Args:
        class_name (str): The name of the criterion class to use.
        **kwargs: Additional keyword arguments to pass to the criterion.

    Returns:
        torch.nn.Module: The initialized criterion.
    """
    return getattr(torch.nn, class_name)(**kwargs)
