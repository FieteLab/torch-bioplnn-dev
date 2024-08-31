import os
import sys
from traceback import print_exc
from typing import Optional

import hydra
import torch
from addict import Dict as AttrDict
from classifiers import ImageClassifier
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import wandb
from bioplnn.utils import get_image_classification_dataloaders, manual_seed, pass_fn


def initialize_optimizer(
    model_parameters, fn, lr, momentum=0.9, beta1=0.9, beta2=0.9, **kwargs
) -> torch.optim.Optimizer:
    if fn == "sgd":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            **kwargs,
        )
    elif fn == "adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=lr,
            betas=(beta1, beta2),
            **kwargs,
        )
    elif fn == "adamw":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=lr,
            betas=(beta1, beta2),
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Optimizer {fn} not implemented")

    return optimizer


def initialize_scheduler(optimizer, fn, max_lr, total_steps):
    if fn is None:
        scheduler = None
    if fn == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
        )
    else:
        raise NotImplementedError(f"Scheduler {fn} not implemented")

    return scheduler


def initialize_criterion(fn: str) -> torch.nn.Module:
    if fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {fn} not implemented")

    return criterion


def train_epoch(
    config: AttrDict,
    model: ImageClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    global_step: int,
    device: torch.device,
) -> tuple[float, float]:
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
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"),
        disable=not config.tqdm,
    )
    for i, (image, labels) in enumerate(iterable=bar):
        image = image.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(
            x=image,
            num_steps=config.train.num_steps,
            loss_all_timesteps=config.criterion.all_timesteps,
        )
        if config.criterion.all_timesteps:
            losses = []
            for output in outputs:
                losses.append(criterion(output, labels))
            loss = sum(losses) / len(losses)
            outputs = outputs[-1]
        else:
            loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_(
            model.parameters(), config.train.grad_clip.value, foreach=config.foreach
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Update statistics
        train_loss += loss.item()
        running_loss += loss.item()

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


def val_epoch(
    config: AttrDict,
    model: ImageClassifier,
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
        tuple: A tuple containing the val loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for image, labels in val_loader:
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                x=image,
                num_steps=config.train.num_steps,
                loss_all_timesteps=config.criterion.all_timesteps,
            )
            if config.criterion.all_timesteps:
                losses = []
                for output in outputs:
                    losses.append(criterion(output, labels))
                loss = sum(losses) / len(losses)
                outputs = outputs[-1]
            else:
                loss = criterion(outputs, labels)

            # Update statistics
            val_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            val_correct += correct
            val_total += len(labels)

    # Calculate average val loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def train(config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """

    config = OmegaConf.to_container(config, resolve=True)
    # print(yaml.dump(config))
    config = AttrDict(config)

    # Initialize Weights & Biases
    wandb.require("core")
    wandb.init(
        config=config,
        settings=wandb.Settings(start_method="thread"),
        **config.wandb,
    )
    global_step = 0

    if config.wandb.mode != "disabled":
        checkpoint_dir = os.path.join(config.checkpoint.root, wandb.run.name)
    else:
        checkpoint_dir = os.path.join(config.checkpoint.root, config.checkpoint.run)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set the random seed
    if config.seed is not None:
        manual_seed(config.seed)

    # Set the matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Get device and initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(**config.model).to(device)

    # Compile the model if requested
    model = torch.compile(model, **config.compile)

    # Initialize the optimizer
    optimizer = initialize_optimizer(model.parameters(), **config.optimizer)

    # Initialize the loss function
    criterion = initialize_criterion(config.criterion.fn)

    # Get the data loaders
    train_loader, val_loader = get_image_classification_dataloaders(
        **config.data,
        resolution=config.model.rnn_kwargs.input_size,
        seed=config.seed,
    )

    # Initialize the learning rate scheduler
    scheduler = initialize_scheduler(
        optimizer,
        fn=config.scheduler.fn,
        max_lr=config.optimizer.lr,
        total_steps=len(train_loader) * config.train.epochs,
    )

    for epoch in range(config.train.epochs):
        wandb.log(dict(epoch=epoch), step=global_step)
        # Train the model
        train_loss, train_acc, global_step = train_epoch(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            epoch,
            global_step,
            device,
        )
        wandb.log(dict(train_loss=train_loss, train_acc=train_acc), step=global_step)

        # Evaluate the model on the validation set
        val_loss, val_acc = val_epoch(config, model, criterion, val_loader, device)
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
        link_path = os.path.abspath(os.path.join(checkpoint_dir, "checkpoint.pt"))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
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
    config_path="/om2/user/valmiki/bioplnn/config/ei_crnn",
    config_name="config",
)
def main(config: DictConfig):
    try:
        train(config)
    except Exception as e:
        if config.debug_level > 1:
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
