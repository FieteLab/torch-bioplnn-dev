import os
import sys
from traceback import print_exc

import hydra
import torch
import wandb
import yaml
from addict import Dict as AttrDict
from bioplnn.models.topography import TopographicalRNN
from bioplnn.optimizers import SparseSGD
from bioplnn.utils import (
    clip_grad_norm_,
    clip_grad_pass_,
    clip_grad_value_,
    get_benchmark_dataloaders,
)
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def train_epoch(
    config: AttrDict,
    model: TopographicalRNN,
    optimizer: torch.optim.Optimizer,
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
        model (TopographicalRNN): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
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
        clip_grad_ = clip_grad_pass_
    elif config.train.grad_clip.type == "norm":
        clip_grad_ = clip_grad_norm_
    elif config.train.grad_clip.type == "value":
        clip_grad_ = clip_grad_value_
    else:
        raise NotImplementedError(
            f"Gradient clipping type {config.train.grad_clip.type} not implemented"
        )

    bar = tqdm(
        train_loader,
        desc=(f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"),
        disable=not config.tqdm,
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
        clip_grad_(
            model.parameters(), config.train.grad_clip.value, foreach=config.foreach
        )
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
            wandb.log(dict(running_loss=running_loss, running_acc=running_acc))
            bar.set_description(
                f"Training | Epoch: {epoch} | "
                f"Loss: {running_loss:.4f} | "
                f"Acc: {running_acc:.2%}"
            )
            running_loss = 0
            running_correct = 0
            running_total = 0

        global_step += 1

    # Calculate average training loss and accuracy
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    return train_loss, train_acc, global_step


def eval_epoch(
    model: TopographicalRNN,
    criterion: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    global_step: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        model (TopographicalRNN): The model to be evaluated.
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

    wandb.log(dict(test_loss=test_loss, test_acc=test_acc))

    return test_loss, test_acc


def train(config: DictConfig) -> None:
    """
    Train the model using the provided configuration.

    Args:
        config (AttrDict): Configuration parameters.
    """
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    config = OmegaConf.to_container(config, resolve=True)
    print(yaml.dump(config))
    config = AttrDict(config)

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
    wandb.init(config=config, **config.wandb)
    global_step = 0

    for epoch in range(config.train.epochs):
        wandb.log(dict(epoch=epoch), step=global_step)
        # Train the model
        train_loss, train_acc, global_step = train_epoch(
            config,
            model,
            optimizer,
            criterion,
            train_loader,
            epoch,
            global_step,
            device,
        )
        wandb.log(dict(train_loss=train_loss, train_acc=train_acc), step=global_step)

        # Evaluate the model on the test set
        test_loss, test_acc = eval_epoch(model, criterion, val_loader, epoch, device)

        wandb.log(dict(test_loss=test_loss, test_acc=test_acc), step=global_step)

        if config.visualize.enable:
            images, _ = next(iter(val_loader))
            images = images.to(device)
            if config.visualize.save_path is not None:
                save_path = (
                    f"{os.path.splitext(config.visualize.save_path)[0]}_{epoch}.gif"
                )
            _, activations = model(images, return_activations=True)
            model.visualize(activations, save_path)

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
