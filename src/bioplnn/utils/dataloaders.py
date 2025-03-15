from typing import Optional

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from bioplnn.utils.torch import manual_seed


def rescale_to_range(x, old_min=0, old_max=1, new_min=-1, new_max=1):
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def get_qclevr_dataloaders(
    *,
    root: str,
    cue_assets_root: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: Optional[tuple[int, int]] = None,
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
    from bioplnn.datasets import QCLEVRDataset

    transform = [
        transforms.ToTensor(),
        transforms.Lambda(rescale_to_range),
    ]
    if resolution is not None:
        transform.append(
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.NEAREST_EXACT,
            )
        )
    transform = transforms.Compose(transform)

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
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
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
    *,
    root: str,
    resolution: Optional[tuple[int, int]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from bioplnn.datasets import CABCDataset

    transform = [
        transforms.ToTensor(),
        transforms.Lambda(rescale_to_range),
    ]
    if resolution is not None:
        transform.append(
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.NEAREST_EXACT,
            )
        )
    transform = transforms.Compose(transform)

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
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    return train_dataloader, val_dataloader


def get_mazes_dataloaders(
    *,
    root: str,
    resolution: Optional[tuple[int, int]] = None,
    subset: float = 1.0,
    return_metadata: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from bioplnn.datasets import Mazes

    if resolution is not None:
        transform = transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.NEAREST_EXACT,
        )
    else:
        transform = None

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
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    return train_dataloader, val_dataloader


def get_correlated_dots_dataloaders(
    resolution: tuple[int, int] = (128, 128),
    n_frames: int = 10,
    n_dots: int = 100,
    correlation: float = 1.0,
    max_speed: int = 5,
    samples_per_epoch: int = 10000,
    val_samples_per_epoch: int = 2000,
    batch_size: int = 512,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    from bioplnn.datasets.correlated_dots import CorrelatedDots

    train_dataset = CorrelatedDots(
        resolution=resolution,
        n_frames=n_frames,
        n_dots=n_dots,
        correlation=correlation,
        max_speed=max_speed,
        samples_per_epoch=samples_per_epoch,
    )

    val_dataset = CorrelatedDots(
        resolution=resolution,
        n_frames=n_frames,
        n_dots=n_dots,
        correlation=correlation,
        max_speed=max_speed,
        samples_per_epoch=val_samples_per_epoch,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=manual_seed if seed is not None else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    return train_dataloader, val_dataloader


def _image_classification_dataloaders(
    dataset: str,
    root: str,
    resolution: Optional[tuple[int, int]] = None,
    v1: bool = False,
    retina_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
):
    from torchvision.datasets import CIFAR10, CIFAR100, MNIST

    from bioplnn.datasets.v1 import CIFAR10_V1, CIFAR100_V1, MNIST_V1

    resize = (
        [
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.NEAREST_EXACT,
            )
        ]
        if resolution is not None
        else []
    )

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
                T.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
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
        worker_init_fn=(lambda x: manual_seed(x + seed))
        if seed is not None
        else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=shuffle_test,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        worker_init_fn=(lambda x: manual_seed(x + seed))
        if seed is not None
        else None,
        generator=torch.Generator().manual_seed(seed)
        if seed is not None
        else None,
    )

    return train_loader, test_loader


def get_image_classification_dataloaders(
    dataset="mnist",
    root="data",
    resolution: Optional[tuple[int, int]] = None,
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
    resolution: Optional[tuple[int, int]] = None,
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
    resolution: Optional[tuple[int, int]] = None,
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
    resolution: Optional[tuple[int, int]] = None,
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
    root: str = "data",
    resolution: Optional[tuple[int, int]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
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
    root: str = "data",
    resolution: Optional[tuple[int, int]] = None,
    retina_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
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
    root: str = "data",
    resolution: Optional[tuple[int, int]] = None,
    retina_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
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
    root: str = "data",
    resolution: Optional[tuple[int, int]] = None,
    retina_path: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: Optional[int] = None,
    shuffle_test: bool = False,
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
