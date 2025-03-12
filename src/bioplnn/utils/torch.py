import math
import os
import random

import numpy as np
from os import PathLike
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from bioplnn.typing import TensorInitFnType


def manual_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def manual_seed_deterministic(seed: int):
    manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _get_single_activation_class(activation: str | None) -> type[nn.Module]:
    """
    Get a single activation function class from the nn module.

    Args:
        activation: The name of the activation function to get.

    Returns:
        The activation function class.
    """
    if activation is None:
        return nn.Identity
    modules = dir(nn)
    modules_lower = [
        module.lower()
        for module in modules
        if not module.startswith("_") and not module.startswith("__")
    ]
    module_count = modules_lower.count(activation.lower())
    if module_count == 0:
        raise ValueError(f"Activation function {activation} not found.")
    elif module_count == 1:
        module_idx = modules_lower.index(activation.lower())
        return getattr(nn, modules[module_idx])
    else:
        try:
            return getattr(nn, activation)
        except Exception:
            raise ValueError(
                f"Multiple activation functions with the (lowercase)"
                f" name {activation.lower()} found, and the"
                f" provided name {activation} could not be found."
            )


def get_activation_class(
    activation: str | None,
) -> type[nn.Module] | list[type[nn.Module]]:
    """
    Get a single or list of activation function classes from the nn module.

    Args:
        activation: The name of the activation function to get.

    Returns:
        The activation function class.
    """
    if activation is None:
        return nn.Identity
    activations = [act.strip() for act in activation.split(",")]
    if len(activations) == 1:
        return _get_single_activation_class(activations[0])
    else:
        return [_get_single_activation_class(act) for act in activations]


def get_activation(activation: str | None) -> nn.Module:
    """
    Get a single or list of activation function classes from the nn module.

    Args:
        activation: The name of the activation function to get.

    Returns:
        The activation function.
    """
    activation_classes = get_activation_class(activation)
    if isinstance(activation_classes, list):
        return nn.Sequential(*[act() for act in activation_classes])
    else:
        return activation_classes()


def init_tensor(
    init_fn: str | TensorInitFnType, *args, **kwargs
) -> torch.Tensor:
    """
    Initialize a tensor using a function or a string.

    Args:
        init_fn: A function or a string specifying the initialization method.
        *args: Positional arguments for the initialization function.
    """

    if isinstance(init_fn, str):
        match init_fn:
            case "zeros":
                return torch.zeros(*args, **kwargs)
            case "ones":
                return torch.ones(*args, **kwargs)
            case "randn":
                return torch.randn(*args, **kwargs)
            case "rand":
                return torch.rand(*args, **kwargs)
            case _:
                raise ValueError(
                    "Invalid initialization function string. Must be 'zeros', "
                    "'ones', 'randn', or 'rand'."
                )

    try:
        return init_fn(*args, **kwargs)
    except TypeError as e:
        if "device" in kwargs:
            return init_fn(*args, **kwargs).to(kwargs["device"])
        else:
            raise e


def idx_1D_to_2D_tensor(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    Convert a 1D index to a 2D index.

    Args:
        x (torch.Tensor): 1D index.
        m (int): Number of rows.
        n (int): Number of columns.

    Returns:
        torch.Tensor: 2D index.
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D_tensor(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """
    Convert a 2D index to a 1D index.

    Args:
        x (torch.Tensor): 2D index.
        m (int): Number of rows (unused).
        n (int): Number of columns.

    Returns:
        torch.Tensor: 1D index.
    """
    return x[0] * n + x[1]


def print_cuda_mem_stats():
    f, t = torch.cuda.mem_get_info()
    print(f"Free/Total: {f / (1024**3):.2f}GB/{t / (1024**3):.2f}GB")


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        num_params = (
            param._nnz()
            if param.layout
            in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc)
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


def create_random_topographic_hh_connectivity(
    sheet_size: tuple[int, int],
    synapse_std: float,
    synapses_per_neuron: int,
    self_recurrence: bool,
) -> torch.Tensor:
    """
    Generates random connectivity matrices for the TRNN.

    Args:
        sheet_size (tuple[int, int]): Size of the sheet-like topology.
        synapse_std (float): Standard deviation for random synapse initialization.
        synapses_per_neuron (int): Number of synapses per neuron.
        self_recurrence (bool): Whether to include self-recurrent connections.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Connectivity matrices for input-to-hidden and hidden-to-hidden connections.
    """
    # Generate random connectivity for hidden-to-hidden connections
    num_neurons = sheet_size[0] * sheet_size[1]

    idx_1d = torch.arange(num_neurons)
    idx = idx_1D_to_2D_tensor(idx_1d, sheet_size[0], sheet_size[1]).t()

    synapses = (
        torch.randn(num_neurons, 2, synapses_per_neuron) * synapse_std
        + idx.unsqueeze(-1)
    ).long()

    if self_recurrence:
        synapses = torch.cat([synapses, idx.unsqueeze(-1)], dim=2)

    synapses = synapses.clamp(
        torch.zeros(2).view(1, 2, 1),
        torch.tensor((sheet_size[0] - 1, sheet_size[1] - 1)).view(1, 2, 1),
    )
    synapses = idx_2D_to_1D_tensor(
        synapses.transpose(0, 1).flatten(1), sheet_size[0], sheet_size[1]
    ).view(num_neurons, -1)

    synapse_root = idx_1d.unsqueeze(-1).expand(-1, synapses.shape[1])

    indices_hh = torch.stack((synapses, synapse_root)).flatten(1)

    ## He initialization of values (synapses_per_neuron is the fan_in)
    values_hh = torch.randn(indices_hh.shape[1]) * math.sqrt(
        2 / synapses_per_neuron
    )

    connectivity_hh = torch.sparse_coo_tensor(
        indices_hh,
        values_hh,
        (num_neurons, num_neurons),
        check_invariants=True,
    ).coalesce()

    return connectivity_hh


def create_identity_ih_connectivity(
    input_size: int,
    num_neurons: int,
    input_indices: Optional[torch.Tensor | PathLike] = None,
) -> torch.Tensor:
    # Generate identity connectivity for input-to-hidden connections
    indices_ih = torch.stack(
        (
            input_indices
            if input_indices is not None
            else torch.arange(input_size),
            torch.arange(input_size),
        )  # type: ignore
    )

    values_ih = torch.ones(indices_ih.shape[1])

    connectivity_ih = torch.sparse_coo_tensor(
        indices_ih,
        values_ih,
        (num_neurons, input_size),
        check_invariants=True,
    ).coalesce()

    return connectivity_ih
