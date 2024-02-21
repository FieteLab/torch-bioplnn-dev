import os
import glob

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor, Resize
from utils import image2v1


class V1Dataset:
    def prepare(self, image_top_corner, Nx, Ny, retina_radius, layer):
        """
        Read cortex information.
        """
        if len(self.retina_path) == 0:
            return

        if os.path.exists(os.path.join(self.retina_path, f"{layer}.npy")):
            self.retina_indices = np.load(
                os.path.join(self.retina_path, f"{layer}.npy")
            )
            self.Nx = Nx
            self.Ny = Ny
            self.retina_radius = retina_radius
        else:
            mask_path = glob.glob(os.path.join(self.retina_path, "*.npy"))[0]
            self.retina_radius = int(
                float(mask_path.split("retrad=")[1].split("_")[0])
            )
            mask = np.load(mask_path)[0]
            self.retina_indices = mask.nonzero()
            self.Nx, self.Ny = mask.shape
        self.image2v1 = lambda x: image2v1(
            x,
            self.retina_indices,
            image_top_corner,
            Nx,
            Ny,
            retina_radius,
            mode=self.mode,
        )


class CIFAR10_V1(CIFAR10, V1Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        retina_path="connection",
        image_top_corner=(4, 4),
        Nx=150,
        Ny=300,
        retina_radius=80,
        layer="V1_indices",
        mode="vector",
        dual_hemisphere=False,
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.image_top_corrner = image_top_corner
        self.mode = mode
        self.retina_path = retina_path
        self.dual_hemisphere = dual_hemisphere
        self.prepare(image_top_corner, Nx, Ny, retina_radius, layer)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.mode == "original":
            return image, target
        if self.dual_hemisphere:
            C, H, W = image.shape
            v1 = [
                self.image2v1(image[:, :, : W // 2])[0],
                self.image2v1(image[:, :, W // 2 :])[0],
            ]
            v1 = torch.stack(v1)
        else:
            v1, _ = self.image2v1(image)
        return v1, target


class MNIST_V1(MNIST, V1Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        retina_path="connection",
        image_top_corner=(4, 4),
        Nx=150,
        Ny=300,
        retina_radius=80,
        layer="V1_indices",
        mode="vector",
        dual_hemisphere=False,
    ):
        super(MNIST_V1, self).__init__(
            root, train, transform, target_transform, download
        )

        self.image_top_corrner = image_top_corner
        self.mode = mode
        self.retina_path = retina_path
        self.dual_hemisphere = dual_hemisphere
        self.prepare(image_top_corner, Nx, Ny, retina_radius, layer)

    def __getitem__(self, index):
        image, target = super(MNIST_V1, self).__getitem__(index)
        if self.mode == "original":
            return image, target
        if self.dual_hemisphere:
            C, H, W = image.shape
            v1 = [
                self.image2v1(image[:, :, : W // 2])[0],
                self.image2v1(image[:, :, W // 2 :])[0],
            ]
            v1 = torch.stack(v1)
        else:
            v1, _ = self.image2v1(image)
        return v1, target


if __name__ == "__main__":
    # Module Test
    dataset = CIFAR10_V1("../data", retina_path="../connection")
    dataset.__getitem__(0)
    breakpoint()
