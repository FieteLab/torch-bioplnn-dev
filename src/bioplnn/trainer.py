import os

import torch
from tqdm import tqdm

import wandb
from bioplnn.dataset import get_MNIST_V1_dataloaders
from bioplnn.sparse_sgd import SparseSGD
from bioplnn.topography import TopographicalRNN
from bioplnn.utils import AttrDict


def train_iter(
    config, model, optimizer, criterion, train_loader, wandb_log, epoch, device
):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    bar = tqdm(
        train_loader,
        desc=(
            f"Training | Epoch: {epoch} | " f"Loss: {0:.4f} | " f"Acc: {0:.2%}"
        ),
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


def eval_iter(model, criterion, test_loader, wandb_log, epoch, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
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
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total

    wandb_log(dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch))

    return test_loss, test_acc


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopographicalRNN(**config.model).to(device)

    if config.optimizer.fn == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    elif config.optimizer.fn == "sparse_sgd":
        optimizer = SparseSGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optimizer.fn} not implemented"
        )

    if config.criterion == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(
            f"Criterion {config.criterion} not implemented"
        )

    train_loader, test_loader = get_MNIST_V1_dataloaders(
        root=config.data.dir,
        retina_path=config.data.retina_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    if config.wandb:
        wandb.init(project="Cortical RNN", config=config)
        wandb_log = lambda x: wandb.log(x)
    else:
        wandb_log = lambda x: None

    for epoch in range(config.train.epochs):
        # Train the model
        train_loss, train_acc = train_iter(
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
        test_loss, test_acc = eval_iter(
            model, criterion, test_loader, wandb_log, epoch, device
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
            os.path.join(config.train.model_dir, f"model_{epoch}.pt")
        )
        link_path = os.path.abspath(
            os.path.join(config.train.model_dir, "model.pt")
        )
        torch.save(model, file_path)
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
        "--config", type=str, default="config/config_random.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)

    train(config)
