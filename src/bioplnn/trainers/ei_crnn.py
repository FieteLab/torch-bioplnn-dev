import logging
import os
import sys
from traceback import print_exc
from typing import Optional

import hydra
import torch
import wandb
import yaml
from addict import Dict as AttrDict
from bioplnn.datasets.qclevr import get_qclevr_dataloaders
from bioplnn.loss import EDLLoss
from bioplnn.models import Conv2dEIRNN
from bioplnn.utils import clip_grad_norm_, clip_grad_pass_, clip_grad_value_, seed
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

log = logging.getLogger(__name__)


def train_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single training iteration.

    Args:
        config (AttrDict): Configuration parameters.
        model (Conv2dEIRNN): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple[float, float]: A tuple containing the training loss and accuracy.
    """
    if config.train.grad_clip.disable:
        clip_grad_ = clip_grad_pass_
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
    for i, (cue, mixture, labels) in enumerate(iterable=bar):
        cue = cue.to(device)
        mixture = mixture.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(
            cue,
            mixture,
            config.train.num_steps,
            all_timesteps=config.criterion.all_timesteps,
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
            wandb.log(dict(running_loss=running_loss, running_acc=running_acc))
            bar.set_description(
                f"Training | Epoch: {epoch} | "
                f"Loss: {running_loss:.4f} | "
                f"Acc: {running_acc:.2%}"
            )
            running_loss = 0
            running_correct = 0
            running_total = 0

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    wandb.log(dict(train_loss=train_loss, train_acc=train_acc))

    return train_loss, train_acc


def eval_iter(
    config: AttrDict,
    model: Conv2dEIRNN,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        model (Conv2dEIRNN): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The test data loader.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the test loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for cue, mixture, labels in val_loader:
            cue = cue.to(device)
            mixture = mixture.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(
                cue,
                mixture,
                config.train.num_steps,
                all_timesteps=config.criterion.all_timesteps,
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
            test_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            test_correct += correct
            test_total += len(labels)

    # Calculate average test loss and accuracy
    test_loss /= len(val_loader)
    test_acc = test_correct / test_total

    wandb.log(dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch))

    return test_loss, test_acc


def train(config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (dict): Configuration parameters.
    """

    config = OmegaConf.to_container(config, resolve=True)
    print(yaml.dump(config))
    config = AttrDict(config)

    # Set the random seed
    if config.seed is not None:
        seed(config.seed)

    # Set the matmul precision
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Get device and initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv2dEIRNN(**config.model).to(device)

    # Compile the model if requested
    model = torch.compile(
        model,
        fullgraph=config.compile.fullgraph,
        dynamic=config.compile.dynamic,
        backend=config.compile.backend,
        mode=config.compile.mode,
        disable=config.compile.disable,
    )

    # Initialize the optimizer
    if config.optimizer.fn == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    elif config.optimizer.fn == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    elif config.optimizer.fn == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer.fn} not implemented")

    # Initialize the loss function
    if config.criterion.fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.criterion.fn == "edl":
        criterion = EDLLoss(num_classes=config.model.num_classes)
    else:
        raise NotImplementedError(f"Criterion {config.criterion.fn} not implemented")

    # Get the data loaders
    train_loader, val_loader = get_qclevr_dataloaders(
        data_root=config.data.root,
        assets_path=config.data.assets_path,
        train_batch_size=config.data.batch_size,
        val_batch_size=config.data.val_batch_size,
        resolution=config.model.input_size,
        holdout=config.data.holdout,
        mode=config.data.mode,
        primitive=config.data.primitive,
        num_workers=config.data.num_workers,
        seed=config.seed,
    )

    # Initialize the learning rate scheduler
    if config.scheduler.fn is None:
        scheduler = None
    if config.scheduler.fn == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.optimizer.lr,
            total_steps=config.train.epochs * len(train_loader),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler.fn} not implemented")

    # Initialize Weights & Biases
    if config.wandb.mode != "disabled":
        wandb.require("core")
        wandb.init(
            **config.wandb,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
        checkpoint_dir = os.path.join(config.checkpoint.root, wandb.run.name)
    else:
        checkpoint_dir = os.path.join(config.checkpoint.root, "test")

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.train.epochs):
        # Train the model
        train_loss, train_acc = train_iter(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            epoch,
            device,
        )

        # Evaluate the model on the validation set
        test_loss, test_acc = eval_iter(
            config, model, criterion, val_loader, epoch, device
        )

        # Print the epoch statistics
        print(
            f"Epoch [{epoch}/{config.train.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_acc:.2%}"
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
    config_path="/om2/user/valmiki/bioplnn/config/crnn",
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
