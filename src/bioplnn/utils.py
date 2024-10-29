import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from addict import Dict
from torch import nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class AttrDict(Dict):
    def __missing__(self, key):
        raise KeyError(key)


def pass_fn(*args, **kwargs):
    pass


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
    elif activation == "relu":
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
    if param is None:
        return [None] * n
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


def rescale(x):
    return x * 2 - 1


def get_qclevr_dataloaders(
    root: str,
    cue_assets_root: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: tuple[int, int] = (128, 128),
    mode: str = "color",
    holdout: list = [],
    primitive: bool = True,
    shape_cue_color: str = "orange",
    return_metadata: bool = False,
    use_cache: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from bioplnn.datasets.qclevr import QCLEVRDataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),
            transforms.Resize(resolution),
        ]
    )
    train_dataset = QCLEVRDataset(
        root=root,
        cue_assets_root=cue_assets_root,
        transform=transform,
        split="train",
        mode=mode,
        holdout=holdout,
        primitive=primitive,
        shape_cue_color=shape_cue_color,
        return_metadata=return_metadata,
        use_cache=use_cache,
        num_workers=num_workers,
    )
    val_dataset = QCLEVRDataset(
        root=root,
        cue_assets_root=cue_assets_root,
        transform=transform,
        split="val",
        mode=mode,
        holdout=holdout,
        primitive=primitive,
        shape_cue_color=shape_cue_color,
        return_metadata=return_metadata,
        use_cache=use_cache,
        num_workers=num_workers,
    )
    # test_dataset = QCLEVRDataset(
    #     root=root,
    #     cue_assets_root=cue_assets_root,
    #     transform=transform,
    #     split="test",
    #     mode=mode,
    #     holdout=holdout,
    #     primitive=primitive,
    #     shape_cue_color=shape_cue_color,
    #     return_metadata=return_metadata,
    #     use_cache=use_cache,
    #     num_workers=num_workers,
    # )
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
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=val_batch_size,
    #     shuffle=shuffle_test,
    #     num_workers=num_workers,
    #     pin_memory=torch.cuda.is_available(),
    #     worker_init_fn=manual_seed if seed is not None else None,
    #     generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    # )
    return train_dataloader, val_dataloader


def get_cabc_dataloaders(
    root: str,
    resolution: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from bioplnn.datasets.cabc import CABCDataset

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
        ]
    )
    train_dataset = CABCDataset(
        root=root,
        transform=transform,
        train=True,
    )
    val_dataset = CABCDataset(
        root=root,
        transform=transform,
        train=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    return train_dataloader, val_dataloader


def get_mazes_dataloaders(
    root: str,
    resolution: Optional[tuple[int, int]] = None,
    subset: float = 1.0,
    return_metadata: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from bioplnn.datasets.mazes import Mazes

    if resolution is not None:
        transform = transforms.Resize(
            size=resolution,
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )
    else:
        transform = nn.Identity()

    train_dataset = Mazes(
        root=root,
        train=True,
        subset=subset,
        return_metadata=return_metadata,
        transform=transform,
    )
    val_dataset = Mazes(
        root=root,
        train=False,
        subset=subset,
        return_metadata=return_metadata,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )

    return train_dataloader, val_dataloader


def _image_classification_dataloaders(
    dataset,
    root,
    resolution=None,
    v1=False,
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST

    from bioplnn.datasets.v1 import CIFAR10_V1, CIFAR100_V1, MNIST_V1

    resize = [transforms.Resize(resolution)] if resolution is not None else []

    if dataset == "mnist":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                *resize,
            ]
        )
        if v1:
            dataset_cls = MNIST_V1
        else:
            dataset_cls = MNIST
    elif dataset == "cifar10":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                *resize,
            ]
        )
        if v1:
            dataset_cls = CIFAR10_V1
        else:
            dataset_cls = CIFAR10
    elif dataset == "cifar100":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                *resize,
            ]
        )
        if v1:
            dataset_cls = CIFAR100_V1
        else:
            dataset_cls = CIFAR100
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    kwargs = {
        "root": root,
        "download": True,
        "transform": transform,
    }
    if v1:
        kwargs = kwargs | {"retina_path": retina_path}

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
        shuffle=shuffle_test,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=(lambda x: manual_seed(x + seed)) if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )

    return train_loader, test_loader


def get_image_classification_dataloaders(
    dataset="mnist",
    root="data",
    resolution=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset=dataset,
        root=root,
        resolution=resolution,
        v1=False,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_v1_dataloaders(
    dataset="mnist",
    root="data",
    resolution=None,
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset=dataset,
        root=root,
        resolution=resolution,
        v1=True,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_mnist_dataloaders(
    root="data",
    resolution=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="mnist",
        root=root,
        resolution=resolution,
        v1=False,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_cifar10_dataloaders(
    root="data",
    resolution=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="cifar10",
        root=root,
        resolution=resolution,
        v1=False,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_cifar100_dataloaders(
    root="data",
    resolution=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="cifar100",
        root=root,
        resolution=resolution,
        v1=False,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_mnist_v1_dataloaders(
    root="data",
    resolution=None,
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="mnist",
        root=root,
        resolution=resolution,
        retina_path=retina_path,
        v1=True,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_cifar10_v1_dataloaders(
    root="data",
    resolution=None,
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="cifar10",
        root=root,
        resolution=resolution,
        v1=True,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )


def get_cifar100_v1_dataloaders(
    root="data",
    resolution=None,
    retina_path=None,
    batch_size=16,
    num_workers=0,
    seed=None,
    shuffle_test=False,
):
    return _image_classification_dataloaders(
        dataset="cifar100",
        root=root,
        resolution=resolution,
        v1=True,
        retina_path=retina_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle_test=shuffle_test,
    )
