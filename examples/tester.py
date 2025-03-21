import os
import sys
from traceback import print_exc
from typing import Any

import hydra
import torch
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from bioplnn.utils import (
    initialize_criterion,
    initialize_dataloader,
    initialize_model,
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
    model: nn.Module,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        config (AttrDict): The experiment configuration.
        model (nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the val loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    bar = tqdm(
        test_loader,
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
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            labels = labels.to(device)
            # Forward pass
            outputs = model(
                x=x,
                num_steps=config.train.num_steps,
                loss_all_timesteps=config.criterion.all_timesteps,
                return_activations=False,
            )
            if config.criterion.all_timesteps:
                logits = outputs.permute(1, 2, 0)
                labels = labels.unsqueeze(-1).expand(-1, outputs.shape[0])
            else:
                logits = outputs

            # Compute the loss
            loss = criterion(logits, labels)

            # Update statistics
            val_loss += loss.item()
            if config.criterion.all_timesteps:
                predicted = outputs[-1].argmax(-1)
            else:
                predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            val_correct += correct
            val_total += len(labels)

    # Calculate average val loss and accuracy
    val_loss /= len(test_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def test(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    if config.wandb.group is not None and not config.wandb.group.endswith(
        "test"
    ):
        config.wandb.group = f"{config.wandb.group}_test"
    wandb.init(
        config=config,
        settings=wandb.Settings(start_method="thread"),
        **config.wandb,
    )

    # Initialize model and load checkpoint
    model = initialize_model(
        dataset=config.data.dataset, config=config.model
    ).to(device)

    checkpoint_path = os.path.join(
        config.checkpoint.root,
        config.checkpoint.run,
        (
            "checkpoint.pt"
            if config.checkpoint.epoch is None
            else f"checkpoint_{config.checkpoint.epoch}.pt"
        ),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval()

    # Initialize the loss function
    criterion = initialize_criterion(config.criterion.fn)

    # Initialize Test Dataloader
    _, test_loader = initialize_dataloader(
        **config.data.dataset, seed=config.seed
    )

    # Compute loss and accuracy on test data
    loss, acc = _test(config, model, criterion, test_loader, device)

    # Record results
    print(f"Loss: {loss}, Acc: {acc}")
    wandb.log(dict(loss=loss, acc=acc))


@hydra.main(
    version_base=None,
    config_path="/om2/user/valmiki/bioplnn/config",
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
        wandb.log(dict(error=str(e)))
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()


# python -u src/bioplnn/trainers/ei_tester.py data.mode=every data.holdout=[] model.modulation_type=ag checkpoint.run=skilled-spaceship-116
