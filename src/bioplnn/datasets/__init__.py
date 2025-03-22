from .cabc import CABCDataset
from .cocodots import CocoDots, make_stimulus
from .correlated_dots import CorrelatedDots
from .mazes import Mazes
from .qclevr import QCLEVRDataset
from .v1 import CIFAR10_V1, CIFAR100_V1, MNIST_V1

__all__ = [
    "Mazes",
    "CIFAR10_V1",
    "CIFAR100_V1",
    "MNIST_V1",
    "CABCDataset",
    "QCLEVRDataset",
    "CocoDots",
    "make_stimulus",
    "CorrelatedDots",
]
