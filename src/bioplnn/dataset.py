import os
import glob

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor, Resize
from .utils import image2v1, flatten_indices


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
        ret = torch.zeros(image.shape[0], self.Nx * self.Ny).to(image.device)
        ret[:, self.flat_indices] = v1
        return ret, target


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
        ret = torch.zeros(self.Nx * self.Ny).to(image.device)
        ret[self.flat_indices] = v1
        return ret, target


if __name__ == "__main__":
    # Module Test
    dataset = CIFAR10_V1("../data", retina_path="../connection/V1_indices.npy")
    dataset.__getitem__(0)
    breakpoint()
