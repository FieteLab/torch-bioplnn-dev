import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, MNIST

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


def get_dataloaders(
    dataset="mnist",
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    retina_path_arg = dict()
    if dataset in ["mnist", "mnist_v1"]:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if dataset == "mnist":
            dataset = MNIST
        else:
            dataset = MNIST_V1
            retina_path_arg = {"retina_path": retina_path}
    elif dataset in ["cifar10", "cifar10_v1"]:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                ),
            ]
        )
        if dataset == "cifar10":
            dataset = CIFAR10
        else:
            dataset = CIFAR10_V1
            retina_path_arg = {"retina_path": retina_path}
    elif dataset in ["cifar100", "cifar100_v1"]:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        if dataset == "cifar100":
            dataset = CIFAR100
        else:
            dataset = CIFAR100_V1
            retina_path_arg = {"retina_path": retina_path}
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    # Load the MNIST dataset

    train_set = dataset(
        root=root,
        train=True,
        download=True,
        transform=transform,
        **retina_path_arg,
    )

    # Load the MNIST test dataset
    test_set = dataset(
        root="./data",
        train=False,
        transform=transform,
        download=True,
        **retina_path_arg,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_set,
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
