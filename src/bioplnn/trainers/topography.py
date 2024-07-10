import os
from typing import Callable

import torch
from addict import Dict as AttrDict
from numpy import clip
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from tqdm import tqdm

import wandb
from bioplnn.models.topography import TopographicalRNN
from bioplnn.sparse_sgd import SparseSGD
from bioplnn.utils import get_benchmark_dataloaders


def train_epoch(
    config: AttrDict,
    model: TopographicalRNN,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single training iteration.

    Args:
        config (AttrDict): Configuration parameters.
        model (TopographicalRNN): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        wandb_log (function): Function to log training statistics to Weights & Biases.
        epoch (int): The current epoch number.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the training loss and accuracy.
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    if config.train.grad_clip.disable:
        clip_grad_fn_ = lambda x: None
    elif config.train.grad_clip.type == "norm":
        clip_grad_fn_ = lambda x: clip_grad_norm_(
            x, config.train.grad_clip.value, foreach=False
        )
    elif config.train.grad_clip.type == "value":
        clip_grad_fn_ = lambda x: clip_grad_value_(
            x, config.train.grad_clip.value, foreach=False
        )
    else:
        raise NotImplementedError(
            f"Gradient clipping type {config.train.grad_clip.type} not implemented"
        )

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"),
    )
    for i, (images, labels) in enumerate(bar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_fn_(model.parameters())
        optimizer.step()

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
            wandb_log(dict(running_loss=running_loss, running_acc=running_acc))
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

    wandb_log(dict(train_loss=train_loss, train_acc=train_acc))

    return train_loss, train_acc


def eval_epoch(
    model: TopographicalRNN,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    wandb_log: Callable[[dict[str, float, int]], None],
    epoch: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        model (TopographicalRNN): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The test data loader.
        wandb_log (function): Function to log evaluation statistics to Weights & Biases.
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
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
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

    wandb_log(dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch))

    return test_loss, test_acc


def train(config: AttrDict) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (AttrDict): Configuration parameters.
    """
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopographicalRNN(**config.model).to(device)

    # Initialize the optimizer
    if config.optimizer.fn == "sgd":
        if config.model.sparse_format in ("coo", "csr"):
            raise ValueError(
                "sgd is not supported with coo or csr: Use sparse_sgd instead"
            )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    elif config.optimizer.fn == "adam":
        if config.model.sparse_format in ("coo", "csr"):
            raise ValueError(
                "adam is not supported with coo or csr: Use sparse_sgd instead"
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    elif config.optimizer.fn == "adamw":
        if config.model.sparse_format in ("coo", "csr"):
            raise ValueError(
                "adam is not supported with coo or csr: Use sparse_sgd instead"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
        )
    elif config.optimizer.fn == "sparse_sgd":
        if config.model.mm_function == "torch_sparse":
            raise ValueError("sparse_sgd is not supported with torch_sparse")
        optimizer = SparseSGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer.fn} not implemented")

    # Initialize the loss function
    if config.criterion == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {config.criterion} not implemented")

    # Get the data loaders
    train_loader, val_loader = get_benchmark_dataloaders(
        dataset=config.data.dataset,
        root=config.data.root,
        retina_path=config.data.retina_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Initialize Weights & Biases
    if config.wandb:
        wandb.init(project="Cortical RNN", config=config)
        wandb_log = lambda x: wandb.log(x)
    else:
        wandb_log = lambda x: None

    for epoch in range(config.train.epochs):
        # Train the model
        train_loss, train_acc = train_epoch(
            config,
            model,
            optimizer,
            criterion,
            train_loader,
            wandb_log,
            epoch,
            device,
        )

        # Evaluate the model on the test set
        test_loss, test_acc = eval_epoch(
            model, criterion, val_loader, wandb_log, epoch, device
        )

        if not config.visualize.disable:
            images, _ = next(iter(val_loader))
            images = images.to(device)
            save_path = None
            if config.visualize.save_path is not None:
                save_path = (
                    f"{os.path.splitext(config.visualize.save_path)[0]}_{epoch}.gif"
                )
            model(
                images,
                visualize=True,
                visualization_save_path=save_path,
                visualization_fps=config.visualize.fps,
                visualization_frames=config.visualize.frames,
            )

        # Print the epoch statistics
        print(
            f"Epoch [{epoch}/{config.train.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_acc:.2%}"
        )

        # Save Model
        file_path = os.path.abspath(
            os.path.join(config.train.checkpoint_dir, f"checkpoint_{epoch}.pt")
        )
        link_path = os.path.abspath(
            os.path.join(config.train.checkpoint_dir, "checkpoint.pt")
        )
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


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/config_topography_random.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    train(config)
