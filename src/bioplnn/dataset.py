import os
import glob

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from bioplnn.utils import image2v1, flatten_indices


class V1Dataset:
    def prepare(self, image_top_corner, Nx, Ny, retina_radius):
        """
        Read cortex information.
        """
        self.retina_indices = np.load(self.retina_path)
        self.flat_indices = torch.tensor(
            flatten_indices(self.retina_indices, Ny)
        )
        self.Nx = Nx
        self.Ny = Ny
        self.retina_radius = retina_radius

        self.image2v1 = lambda x: image2v1(
            x,
            self.retina_indices,
            image_top_corner,
            Nx,
            Ny,
            retina_radius,
        )


class CIFAR10_V1(CIFAR10, V1Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        retina_path="connection/V1_indices.npy",
        image_top_corner=(4, 4),
        Nx=150,
        Ny=300,
        retina_radius=80,
        dual_hemisphere=False,
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.image_top_corrner = image_top_corner
        self.retina_path = retina_path
        self.dual_hemisphere = dual_hemisphere
        self.prepare(image_top_corner, Nx, Ny, retina_radius)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        v1 = self.image2v1(image)
        return v1, target


class MNIST_V1(MNIST, V1Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        retina_path="connection/V1_indices.npy",
        image_top_corner=(4, 4),
        Nx=150,
        Ny=300,
        retina_radius=80,
        dual_hemisphere=False,
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.image_top_corrner = image_top_corner
        self.retina_path = retina_path
        self.dual_hemisphere = dual_hemisphere
        self.prepare(image_top_corner, Nx, Ny, retina_radius)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        v1 = self.image2v1(image)
        return v1, target


def get_MNIST_V1_dataloaders(
    root="./data",
    retina_path="connection/V1_indices.npy",
    batch_size=16,
    num_workers=0,
):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Load the MNIST dataset
    mnist_train = MNIST_V1(
        root=root,
        train=True,
        download=True,
        transform=transform,
        retina_path=retina_path,
    )

    # Load the MNIST test dataset
    mnist_test = MNIST_V1(
        root="./data",
        train=False,
        transform=transform,
        download=True,
        retina_path=retina_path,
    )

    train_loader = DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=mnist_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Module Test
    dataset = CIFAR10_V1("../data", retina_path="../connection/V1_indices.npy")
    dataset.__getitem__(0)
    breakpoint()
