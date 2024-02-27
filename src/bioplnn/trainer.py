import os

import torch
from tqdm import tqdm

import wandb
from bioplnn.dataset import get_MNIST_V1_dataloaders
from bioplnn.topography import TopographicalRNN


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopographicalRNN(**config).to(device)  # type: ignore
    optimizer = config.optimizer(model.parameters(), lr=config.lr)
    criterion = config.criterion()
    train_loader, test_loader = get_MNIST_V1_dataloaders(
        root=config.data_dir,
        retina_path=config.retina_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    if config.wandb:
        wandb.init(project="Cortical RNN", config=config)

    for epoch in range(config.num_epochs):
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
                f"Training | Epoch: {epoch} | "
                f"Loss: {0:.4f} | "
                f"Acc: {0:.2%}"
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
            if (i + 1) % config.log_freq == 0:
                running_loss /= config.log_freq
                running_acc = running_correct / running_total
                if config.wandb:
                    wandb.log(
                        dict(
                            running_loss=running_loss, running_acc=running_acc
                        )
                    )
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

        if config.wandb:
            wandb.log(dict(train_loss=train_loss, train_acc=train_acc))

        # Evaluate the model on the test set
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
        test_loss /= len(train_loader)
        test_acc = test_correct / test_total

        if config.wandb:
            wandb.log(
                dict(test_loss=test_loss, test_acc=test_acc, epoch=epoch)
            )

        # Print the epoch statistics
        print(
            f"Epoch [{epoch}/{config.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_acc:.2%}"
        )

        # Save Model
        # Save Model
        file_path = os.path.abspath(
            os.path.join(config.model_dir, f"model_{epoch}.pt")
        )
        link_path = os.path.abspath(os.path.join(config.model_dir, "model.pt"))
        torch.save(model, file_path)
        try:
            os.remove(link_path)
        except FileNotFoundError:
            pass
        os.symlink(file_path, link_path)
