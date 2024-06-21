import os
import random
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.interpolate
import torch
import torchvision.transforms as T
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)


# class AttrDict2(SimpleNamespace):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         for k, v in self.__dict__.items():
#             if isinstance(v, dict):
#                 self.__dict__[k] = AttrDict2(v)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_activation_class(activation):
    if activation is None or activation == "identity":
        return nn.Identity
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "softsign":
        return nn.Softsign
    elif activation == "elu":
        return nn.ELU
    elif activation == "selu":
        return nn.SELU
    elif activation == "gelu":
        return nn.GELU
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Activation function {activation} not supported.")


def idx_1D_to_2D(x, m, n):
    """
    Convert a 1D index to a 2D index.

    Args:
        x (torch.Tensor): 1D index.

    Returns:
        torch.Tensor: 2D index.
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D(x, m, n):
    """
    Convert a 2D index to a 1D index.

    Args:
        x (torch.Tensor): 2D index.

    Returns:
        torch.Tensor: 1D index.
    """
    return x[0] * n + x[1]


def print_mem_stats():
    f, t = torch.cuda.mem_get_info()
    print(f"Free/Total: {f/(1024**3):.2f}GB/{t/(1024**3):.2f}GB")


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        num_params = (
            param._nnz()
            if param.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
            else param.numel()
        )
        total_params += num_params
    return total_params


def profile_fn(fn, kwargs, sort_by="cuda_time_total", row_limit=50):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn(kwargs)
    return prof.key_averages.table(sort_by=sort_by, row_limit=row_limit)


def r_theta_mp(data):
    tmp = np.exp(data[0] + 1j * data[1]) - 0.5
    return np.abs(tmp), np.angle(tmp)


def normalize_for_mp(indices, N_x=150, N_y=300, retina_radius=80):
    x, y = indices
    normalized_x = (1 - (x) / retina_radius) * 2.4 - 0.6
    normalized_y = ((y - N_y // 2) / np.sqrt(retina_radius**2.0)) * 3.5
    return normalized_x, normalized_y


def flatten_indices(indices, N_y=300):
    return indices[0] * N_y + indices[1]


def image2v1(
    image,
    retina_indices,
    image_top_corner=(4, 4),
    N_x=150,
    N_y=300,
    retina_radius=80,
):
    image_x, image_y = image.shape[1:]  # (C, H, W)
    img_ind = np.zeros((2, image_x, image_y))
    img_ind[0, :, :] = (
        np.tile(0 + np.arange(image_x), (image_y, 1)).T / image_x * image_top_corner[0]
    )
    img_ind[1, :, :] = (
        np.tile(np.arange(image_y) - image_y // 2, (image_x, 1))
        / image_y
        * image_top_corner[1]
        * 2
    )

    flat_img_ind = img_ind.reshape((2, image_x * image_y))

    normed_indices_retina = normalize_for_mp(retina_indices, N_x, N_y, retina_radius)
    r_indices, theta_indices = r_theta_mp(normed_indices_retina)

    v_field_x = r_indices * np.cos(theta_indices)
    v_field_y = r_indices * np.sin(theta_indices)

    device = image.device
    image = image.cpu().numpy()

    if len(image.shape) == 3:
        img_on_vfield = [
            scipy.interpolate.griddata(
                flat_img_ind.T,
                im.flatten(),
                np.array((v_field_x, v_field_y)).T,
            )
            for im in image
        ]
        img_on_vfield = np.stack(img_on_vfield)
    else:
        img_on_vfield = scipy.interpolate.griddata(
            flat_img_ind.T,
            image[0].flatten(),
            np.array((v_field_x, v_field_y)).T,
        )

    img_on_vfield = torch.from_numpy(img_on_vfield).to(device).float()
    img_on_vfield = torch.nan_to_num(img_on_vfield)
    return img_on_vfield


def compact(l):
    return list(filter(None, l))


def rescale(x):
    return x * 2 - 1


def _get_dataloaders(
    dataset="mnist",
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    from bioplnn.datasets import CIFAR10_V1, CIFAR100_V1, MNIST_V1

    retina_path_arg = dict()
    if dataset in ["mnist", "mnist_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if dataset == "mnist":
            dataset = MNIST
        else:
            dataset = MNIST_V1
            retina_path_arg = {"retina_path": retina_path}
    elif dataset in ["cifar10", "cifar10_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        if dataset == "cifar10":
            dataset = CIFAR10
        else:
            dataset = CIFAR10_V1
            retina_path_arg = {"retina_path": retina_path}
    elif dataset in ["cifar100", "cifar100_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
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


def get_mnist_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="mnist",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_cifar10_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="cifar10",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_cifar100_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="cifar100",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_mnist_v1_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="mnist_v1",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_cifar10_v1_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="cifar10_v1",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_cifar100_v1_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return _get_dataloaders(
        dataset="cifar100_v1",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
