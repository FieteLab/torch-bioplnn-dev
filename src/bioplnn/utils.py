import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import transforms

from bioplnn.datasets.qclevr import qCLEVRDataset


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def manual_seed_deterministic(seed):
    manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


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
    elif activation == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Activation function {activation} not supported.")


def pass_fn(*args, **kwargs):
    pass


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


def dict_flatten(d, delimiter=".", key=None):
    key = f"{key}{delimiter}" if key is not None else ""
    non_dicts = {f"{key}{k}": v for k, v in d.items() if not isinstance(v, dict)}
    dicts = {
        f"{key}{k}": v
        for _k, _v in d.items()
        if isinstance(_v, dict)
        for k, v in dict_flatten(_v, delimiter=delimiter, key=_k).items()
    }
    for k in list(dicts.keys()):
        if k in non_dicts:
            raise ValueError(f"Key {k} is used more than once in dict.")
    return non_dicts | dicts


def expand_list(param, n, depth=0):
    inner = param
    for _ in range(depth):
        if not isinstance(inner, (list, tuple)):
            raise ValueError(f"The intermediate depth {depth} is not a list or tuple")
        inner = inner[0]

    if not isinstance(inner, (list, tuple)):
        param = [param] * n

    if len(param) != n:
        raise ValueError(
            "The length of param must equal n if the inner variable at the given depth is already a list or tuple."
        )

    return param


def print_cuda_mem_stats():
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


def get_qclevr_dataloaders(
    data_root: str,
    assets_path: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: tuple[int, int],
    holdout: list = [],
    mode: str = "color",
    primitive: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
):
    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
            transforms.Resize(resolution),
        ]
    )
    train_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="train",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
    )
    val_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="valid",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    return train_dataloader, val_dataloader


def get_benchmark_dataloaders(
    dataset="mnist",
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
):
    from bioplnn.datasets import CIFAR10_V1, CIFAR100_V1, MNIST_V1

    if dataset in ["mnist", "mnist_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        if dataset == "mnist":
            dataset_cls = MNIST
        else:
            dataset_cls = MNIST_V1
    elif dataset in ["cifar10", "cifar10_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        if dataset == "cifar10":
            dataset_cls = CIFAR10
        else:
            dataset_cls = CIFAR10_V1
    elif dataset in ["cifar100", "cifar100_v1"]:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        if dataset == "cifar100":
            dataset_cls = CIFAR100
        else:
            dataset_cls = CIFAR100_V1
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    if dataset.endswith("_v1"):
        kwargs = {
            "root": root,
            "download": True,
            "transform": transform,
            "retina_path": retina_path,
        }
    else:
        kwargs = {
            "root": root,
            "download": True,
            "transform": transform,
        }

    train_set = dataset_cls(train=True, **kwargs)
    test_set = dataset_cls(train=False, **kwargs)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=(lambda x: manual_seed(x + seed)) if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=(lambda x: manual_seed(x + seed)) if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )

    return train_loader, test_loader


def get_mnist_dataloaders(
    root="data",
    retina_path=None,
    batch_size=16,
    num_workers=0,
):
    return get_benchmark_dataloaders(
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
    return get_benchmark_dataloaders(
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
    return get_benchmark_dataloaders(
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
    return get_benchmark_dataloaders(
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
    return get_benchmark_dataloaders(
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
    return get_benchmark_dataloaders(
        dataset="cifar100_v1",
        root=root,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
