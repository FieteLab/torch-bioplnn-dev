import os
import sys
from collections.abc import Callable
from traceback import print_exc
from typing import Any

import hydra
import torch
import wandb
import yaml
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from bioplnn.utils import (
    initialize_criterion,
    initialize_dataloader,
    initialize_model,
    manual_seed,
    manual_seed_deterministic,
)


class AttrDict(Dict):
    """A non-default version of the `addict.Dict` class that raises a `KeyError`
    when a key is not found in the dictionary.

    Args:
        *args: Any positional arguments.
        **kwargs: Any keyword arguments.
    """

    def __missing__(self, key: Any):
        """Override the default behavior of `addict.Dict` to raise a `KeyError`
        when a key is not found in the dictionary.

        Args:
            key: The key that was not found.

        Raises:
            KeyError: Always raised.
        """
        raise KeyError(key)


def _test(
    config: AttrDict,
    model: nn.Module | Callable,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        model (Conv2dEIRNNModulatoryImageClassifier): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The val data loader.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing the val loss and accuracy.
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(
        data_loader,
        desc="Test",
        disable=not config.tqdm,
    )
    with torch.no_grad():
        for i, (x, labels) in enumerate(bar):
            if (
                config.debug_num_batches is not None
                and i >= config.debug_num_batches
            ):
                break

            # Move data to device
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            labels = labels.to(device)
            # Forward pass
            torch.compiler.cudagraph_mark_step_begin()
            outputs = model(x=x, **config.train.forward_kwargs)
            if (
                "loss_all_timesteps" in config.train.forward_kwargs
                and config.train.forward_kwargs.loss_all_timesteps
            ):
                logits = outputs.permute(1, 2, 0)
                loss_labels = labels.unsqueeze(-1).expand(-1, outputs.shape[0])
            else:
                logits = outputs
                loss_labels = labels

            # Compute the loss
            loss = criterion(logits, loss_labels)

            # Update statistics
            loss += loss.item()
            if (
                "loss_all_timesteps" in config.train.forward_kwargs
                and config.train.forward_kwargs.loss_all_timesteps
            ):
                predicted = outputs[-1].argmax(-1)
            else:
                predicted = outputs.argmax(-1)
            correct_cur = (predicted == labels).sum().item()
            correct += correct_cur
            total += len(labels)

    # Calculate average test loss and accuracy
    loss /= len(data_loader)
    acc = correct / total

    return loss, acc


def test(dict_config: DictConfig) -> None:
    """
    Test the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """

    config = OmegaConf.to_container(dict_config, resolve=True)
    config = AttrDict(config)

    # Override parameters
    def parse_overrides(original: Any, overrides: Any) -> Any:
        if isinstance(overrides, dict):
            assert isinstance(original, dict)
            for key, value in overrides.items():
                original[key] = parse_overrides(original[key], value)
        elif isinstance(overrides, list):
            assert isinstance(original, list)
            assert len(original) == len(overrides)
            for i, item in enumerate(original):
                original[i] = parse_overrides(item, overrides[i])
        else:
            assert type(original) is type(overrides)
            original = overrides
        return original

    if "overrides" in config:
        config = parse_overrides(config, config.overrides)

    # Set up debugging
    if config.debug_level > 0:
        print(yaml.dump(config.to_dict()))
        yaml.dump(config.to_dict(), open("examples/config_log.yaml", "w+"))

    if config.debug_level > 1:
        torch.autograd.set_detect_anomaly(True)

    # Set up Weights & Biases
    if config.wandb.group is None:
        config.wandb.group = f"{config.model.class_name}_{config.data.dataset}"

    wandb.init(
        config=config,
        settings=wandb.Settings(start_method="thread"),
        **config.wandb,
    )
    global_step: int = 0

    # Set the random seed
    if config.seed is not None:
        if config.deterministic:
            manual_seed_deterministic(config.seed)
        else:
            manual_seed(config.seed)
    else:
        if config.deterministic:
            raise ValueError(
                "Seed must be provided for deterministic training"
            )

    # Set the matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = initialize_model(**config.model).to(device)

    # Load the model checkpoint if requested
    checkpoint_dir = os.path.join(
        config.checkpoint.root, config.checkpoint.run
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Compile the model if requested
    model = torch.compile(model, **config.compile)

    # Initialize the loss function
    criterion = initialize_criterion(**config.criterion)

    # Get the data loaders
    _, test_loader = initialize_dataloader(seed=config.seed, **config.data)

    loss, acc = _test(
        config=config,
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        device=device,
    )
    wandb.log(dict(test_loss=loss, test_acc=acc), step=global_step)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(__file__), "../config"),
    config_name="config",
)
def main(config: DictConfig):
    try:
        test(config)
    except Exception as e:
        if config.debug_level > 0:
            print_exc(file=sys.stderr)
        else:
            print(e, file=sys.stderr)
        if config.wandb.mode != "disabled":
            wandb.log(dict(error=str(e)))
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
