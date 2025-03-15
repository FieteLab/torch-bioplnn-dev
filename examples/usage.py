import torch
import torchvision

from bioplnn.models import (
    SpatiallyEmbeddedAreaConfig,
    SpatiallyEmbeddedRNN,
)


def example_forward_pass():
    # Define the model
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            in_size=(32, 32),
            in_channels=3,
            out_channels=16,
        )
    ]
    model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs)

    # Define the input (num_steps, batch size, channels, height, width)
    x = torch.randn(10, 16, 3, 32, 32)

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass
    y = model(x, num_steps=10)

    print(y.shape)


def example_backward_pass():
    # Define the model
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            in_size=(32, 32),
            in_channels=3,
            out_channels=16,
        )
    ]
    model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the input (batch size, channels, height, width)
    x = torch.randn(16, 3, 32, 32)

    # Set the model to training mode
    model.train()

    # Perform a forward pass
    y = model(x, num_steps=10)

    # Perform a backward pass
    loss = loss_fn(y, x)
    loss.backward()

    # Update the model parameters
    optimizer.step()


def example_training_loop():
    # Define the model
    area_configs = [
        SpatiallyEmbeddedAreaConfig(
            in_size=(32, 32),
            in_channels=3,
            out_channels=16,
        )
    ]
    model = SpatiallyEmbeddedRNN(num_areas=1, area_configs=area_configs)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the dataset
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True
    )

    # Define the training loop
    for epoch in range(10):
        for batch in dataloader:
            x, y = batch
            y = model(x, num_steps=10)
            loss = loss_fn(y, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    example_forward_pass()
    example_backward_pass()
    example_training_loop()
