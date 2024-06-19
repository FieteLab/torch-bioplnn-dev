import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

from bioplnn.utils import flatten_indices, image2v1


class V1Dataset:
    def prepare(self, retina_path, image_top_corner, Nx, Ny, retina_radius):
        """
        Read cortex information.
        """
        self.retina_indices = np.load(retina_path)
        self.flat_indices = torch.tensor(flatten_indices(self.retina_indices, Ny))
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
        self.dual_hemisphere = dual_hemisphere
        self.prepare(retina_path, image_top_corner, Nx, Ny, retina_radius)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        v1 = self.image2v1(image)
        return v1, target


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
        self.dual_hemisphere = dual_hemisphere
        self.prepare(retina_path, image_top_corner, Nx, Ny, retina_radius)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        v1 = self.image2v1(image)
        return v1, target


class CIFAR100_V1(CIFAR100, V1Dataset):
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
        self.dual_hemisphere = dual_hemisphere
        self.prepare(retina_path, image_top_corner, Nx, Ny, retina_radius)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        v1 = self.image2v1(image)
        return v1, target


if __name__ == "__main__":
    # Module Test
    dataset = CIFAR10_V1("../data", retina_path="../connection/V1_indices.npy")
    dataset.__getitem__(0)
    breakpoint()
