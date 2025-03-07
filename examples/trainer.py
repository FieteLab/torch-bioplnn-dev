import os
import sys
from collections.abc import Callable
from traceback import print_exc
from typing import Optional

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

import wandb
from bioplnn.utils import (
    AttrDict,
    initialize_criterion,
    initialize_dataloader,
    initialize_model,
    initialize_optimizer,
    initialize_scheduler,
    manual_seed,
    manual_seed_deterministic,
    pass_fn,
)


def _train(
    config: AttrDict,
    model: nn.Module | Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    global_step: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """
    Perform a single training iteration.

    Args:
        config (AttrDict): Configuration parameters.
        model (Conv2dEIRNNModulatoryImageClassifier): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing the training loss and accuracy.
    """
    if not config.train.grad_clip.enable:
        clip_grad_ = pass_fn
    elif config.train.grad_clip.type == "norm":
        clip_grad_ = clip_grad_norm_
    elif config.train.grad_clip.type == "value":
        clip_grad_ = clip_grad_value_
    else:
        raise NotImplementedError(
            f"Gradient clipping type {config.train.grad_clip.type} not implemented"
        )

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | Loss: {0:.4f} | Acc: {0:.2%}"),
        disable=not config.tqdm,
    )
    for i, (x, labels) in enumerate(iterable=bar):
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
        outputs = model(
            x=x,
            num_steps=config.train.num_steps,
            loss_all_timesteps=config.train.loss_all_timesteps,
            return_activations=False,
        )
        if config.train.loss_all_timesteps:
            logits = outputs.permute(1, 2, 0)
            loss_labels = labels.unsqueeze(-1).expand(-1, outputs.shape[0])
        else:
            logits = outputs
            loss_labels = labels

        # Compute the loss
        loss = criterion(logits, loss_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_(model.parameters(), config.train.grad_clip.value)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Update statistics
        train_loss += loss.item()
        running_loss += loss.item()
        if config.train.loss_all_timesteps:
            predicted = outputs[-1].argmax(-1)
        else:
            predicted = outputs.argmax(-1)
        correct = (predicted == labels).sum().item()
        train_correct += correct
        running_correct += correct
        train_total += len(labels)
        running_total += len(labels)

        # Log statistics
        if (i + 1) % config.train.log_freq == 0:
            running_loss /= config.train.log_freq
            running_acc = running_correct / running_total
            wandb.log(
                dict(running_loss=running_loss, running_acc=running_acc),
                step=global_step,
            )
            bar.set_description(
                f"Training | Epoch: {epoch} | "
                f"Loss: {running_loss:.4f} | "
                f"Acc: {running_acc:.2%}"
            )
            running_loss = 0
            running_correct = 0
            running_total = 0

        global_step += len(labels)

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    return train_loss, train_acc, global_step


def _validate(
    config: AttrDict,
    model: nn.Module | Callable,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
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
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    bar = tqdm(
        val_loader,
        desc="Validation",
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
            outputs = model(
                x=x,
                num_steps=config.train.num_steps,
                loss_all_timesteps=config.train.loss_all_timesteps,
                return_activations=False,
            )
            if config.train.loss_all_timesteps:
                logits = outputs.permute(1, 2, 0)
                loss_labels = labels.unsqueeze(-1).expand(-1, outputs.shape[0])
            else:
                logits = outputs
                loss_labels = labels

            # Compute the loss
            loss = criterion(logits, loss_labels)

            # Update statistics
            val_loss += loss.item()
            if config.train.loss_all_timesteps:
                predicted = outputs[-1].argmax(-1)
            else:
                predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            val_correct += correct
            val_total += len(labels)

    # Calculate average val loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def train(dict_config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """

    config = OmegaConf.to_container(dict_config, resolve=True)
    config = AttrDict(config)

    # Override parameters
    try:
        layer_one_in_channels_override = config.layer_one_in_channels_override
    except KeyError:
        pass
    else:
        if layer_one_in_channels_override is not None:
            config.model.rnn_kwargs.layer_kwargs[
                0
            ].in_channels = config.layer_one_in_channels_override

    if config.debug_level > 0:
        print(yaml.dump(config.to_dict()))
        yaml.dump(config.to_dict(), open("examples/config_log.yaml", "w+"))

    if config.debug_level > 1:
        torch.autograd.set_detect_anomaly(True)

    if config.wandb.group is None:
        config.wandb.group = f"{config.model.class_name}_{config.data.dataset}"

    # Initialize Weights & Biases
    wandb.init(
        config=config,
        settings=wandb.Settings(start_method="thread"),
        **config.wandb,
    )
    global_step: int = 0

    if config.wandb.mode != "disabled":
        checkpoint_dir = os.path.join(config.checkpoint.root, wandb.run.name)  # type: ignore
    else:
        checkpoint_dir = os.path.join(
            config.checkpoint.root, config.checkpoint.run
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    # Compile the model if requested
    model = torch.compile(model, **config.compile)

    # Initialize the optimizer
    optimizer = initialize_optimizer(
        model_parameters=model.parameters(), **config.optimizer
    )

    # Initialize the loss function
    criterion = initialize_criterion(**config.criterion)

    # Get the data loaders
    train_loader, val_loader = initialize_dataloader(
        seed=config.seed, **config.data
    )

    # Initialize the learning rate scheduler
    if "scheduler" in config and config.scheduler is not None:
        scheduler = initialize_scheduler(
            optimizer=optimizer,
            max_lr=config.optimizer.lr,
            total_steps=len(train_loader) * config.train.epochs,
            **config.scheduler,
        )
    else:
        scheduler = None

    for epoch in range(config.train.epochs):
        print(f"Epoch {epoch}/{config.train.epochs}")
        wandb.log({"epoch": epoch}, step=global_step)
        # Train the model
        train_loss, train_acc, global_step = _train(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            epoch=epoch,
            global_step=global_step,
            device=device,
        )
        wandb.log(
            dict(train_loss=train_loss, train_acc=train_acc), step=global_step
        )

        # Evaluate the model on the validation set
        val_loss, val_acc = _validate(
            config=config,
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
        )
        wandb.log(dict(test_loss=val_loss, test_acc=val_acc), step=global_step)

        # Print the epoch statistics
        print(
            f"Epoch [{epoch}/{config.train.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2%} | "
            f"Test Loss: {val_loss:.4f}, "
            f"Test Accuracy: {val_acc:.2%}"
        )

        # Save the model
        file_path = os.path.abspath(
            os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")
        )
        link_path = os.path.abspath(
            os.path.join(checkpoint_dir, "checkpoint.pt")
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": getattr(
                model, "_orig_mod", model
            ).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)
        try:
            os.remove(link_path)
        except FileNotFoundError:
            pass
        os.symlink(file_path, link_path)


@hydra.main(
    version_base=None,
    config_path="/om2/user/valmiki/bioplnn/config",
    config_name="config",
)
def main(config: DictConfig):
    try:
        train(config)
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
