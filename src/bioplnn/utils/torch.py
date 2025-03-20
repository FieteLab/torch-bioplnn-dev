import math
import os
import random
from collections.abc import Iterable
from os import PathLike
from typing import Any, List, Optional, Type, Union

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from bioplnn.typing import TensorInitFnType


def manual_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def manual_seed_deterministic(seed: int):
    """Set random seeds and configure PyTorch for deterministic execution.

    Args:
        seed (int): The random seed to use.
    """
    manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _get_single_activation_class(
    activation: Union[str, None],
) -> Type[nn.Module]:
    """Get a single activation function class from the nn module.

    Args:
        activation (str, optional): The name of the activation function to get.
            If None, returns nn.Identity. Defaults to None.

    Returns:
        Type[nn.Module]: The activation function class.

    Raises:
        ValueError: If the activation function is not found.
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
    activation: Union[str, None],
) -> Union[Type[nn.Module], List[Type[nn.Module]]]:
    """Get one or more activation function classes.

    If activation is a string with commas, split and get each activation.

    Args:
        activation (str, optional): The name(s) of the activation function(s).
            If None, returns nn.Identity. Defaults to None.

    Returns:
        Union[Type[nn.Module], List[Type[nn.Module]]]: A single activation class
            or a list of activation classes if comma-separated.
    """
    if activation is None:
        return nn.Identity
    activations = [act.strip() for act in activation.split(",")]
    if len(activations) == 1:
        return _get_single_activation_class(activations[0])
    else:
        return [_get_single_activation_class(act) for act in activations]


def get_activation(
    activation: Union[str, None, nn.Module],
) -> nn.Module:
    """Get an initialized activation function module.

    Args:
        activation (Union[str, nn.Module], optional): The name(s) of the
            activation function(s) or an already initialized nn.Module. If the
            latter, the moduleis returned as is. If None, returns nn.Identity().

    Returns:
        nn.Module: The initialized activation function.
    """
    if isinstance(activation, nn.Module):
        return activation
    activation_classes = get_activation_class(activation)
    if isinstance(activation_classes, list):
        return nn.Sequential(*[act() for act in activation_classes])
    else:
        return activation_classes()


def init_tensor(
    init_fn: Union[str, TensorInitFnType], *args, **kwargs
) -> torch.Tensor:
    """Initialize a tensor with a specified initialization function.

    Args:
        init_fn (Union[str, TensorInitFnType]): The initialization function name
            or callable.
        *args: Arguments to pass to the initialization function (usually shape).
        **kwargs: Keyword arguments to pass to the initialization function.

    Returns:
        torch.Tensor: The initialized tensor.

    Raises:
        ValueError: If the initialization function is not supported.
    """

    if isinstance(init_fn, str):
        if init_fn == "zeros":
            return torch.zeros(*args, **kwargs)
        elif init_fn == "ones":
            return torch.ones(*args, **kwargs)
        elif init_fn == "randn":
            return torch.randn(*args, **kwargs)
        elif init_fn == "rand":
            return torch.rand(*args, **kwargs)
        else:
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


def array_from_numpy(
    array: Union[np.ndarray, PathLike, str],
) -> np.ndarray:
    """Create a numpy array from a numpy array or path to file.

    Args:
        array: Numpy array or path to file.

    Returns:
        np.ndarray: The loaded numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    else:
        return np.load(array)


def array_from_df(
    df: Union[pd.DataFrame, PathLike, str],
) -> np.ndarray:
    """Create a numpy array from a dataframe.

    Args:
        df (Union[pd.DataFrame, PathLike, str]): Dataframe or path to file
            containing dataframe.

    Returns:
        The connectivity numpy array.
    """
    if isinstance(df, pd.DataFrame):
        return df.to_numpy()
    else:
        return pd.read_csv(df).to_numpy()


def array_from_tensor(
    tensor: Union[torch.Tensor, PathLike, str],
) -> np.ndarray:
    """Create a numpy array from a numpy array.

    Args:
        tensor (Union[torch.Tensor, PathLike, str]): Torch tensor or path to file.

    Returns:
        np.ndarray: The numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return torch.load(tensor, weights_only=True).numpy()


def tensor_from_df(
    df: Union[pd.DataFrame, PathLike, str],
) -> torch.Tensor:
    """Create a dense tensor from a dataframe.

    Args:
        df (Union[pd.DataFrame, PathLike, str]): Dataframe or path to file
            containing dataframe.

    Returns:
        torch.Tensor: The connectivity tensor.
    """
    if isinstance(df, pd.DataFrame):
        return torch.tensor(df.to_numpy())
    else:
        return torch.tensor(pd.read_csv(df).to_numpy())


def tensor_from_numpy(
    tensor: Union[np.ndarray, PathLike, str],
) -> torch.Tensor:
    """Create a dense tensor from a numpy array.

    Args:
        tensor (np.ndarray): Numpy array.

    Returns:
        torch.Tensor: The sparse tensor.
    """
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    else:
        return torch.tensor(np.load(tensor))


def tensor_from_tensor(
    tensor: Union[torch.Tensor, PathLike, str],
) -> torch.Tensor:
    """Create a tensor from a torch tensor or path to file.

    Args:
        tensor: Torch tensor or path to file.

    Returns:
        The original or loaded tensor.

    Raises:
        ValueError: If the file is not a torch tensor or path to file.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return torch.load(tensor, weights_only=True)


def sparse_tensor_from_df(
    df: Union[pd.DataFrame, PathLike, str],
) -> torch.Tensor:
    """Create a sparse tensor from a dataframe.

    Args:
        df (Union[pd.DataFrame, PathLike, str]): Dataframe or path to file
            containing dataframe.

    Returns:
        torch.Tensor: The connectivity tensor.
    """
    if isinstance(df, pd.DataFrame):
        return torch.tensor(df.to_numpy()).to_sparse()
    else:
        return torch.tensor(pd.read_csv(df).to_numpy()).to_sparse()


def sparse_tensor_from_numpy(
    tensor: Union[np.ndarray, PathLike, str],
) -> torch.Tensor:
    """Create a sparse tensor from a numpy array.

    Args:
        tensor (np.ndarray): Numpy array.

    Returns:
        torch.Tensor: The sparse tensor.
    """
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor).to_sparse()
    else:
        return torch.tensor(np.load(tensor)).to_sparse()


def sparse_tensor_from_scipy(
    array: Union[scipy.sparse.sparray, PathLike, str],
) -> torch.Tensor:
    """Create a sparse tensor from a scipy csr matrix.

    Args:
        tensor (Union[scipy.sparse.coo_matrix, scipy.sparse.csr_matrix, PathLike, str]): Scipy csr matrix.

    Returns:
        torch.Tensor: The sparse tensor.
    """
    if not isinstance(array, scipy.sparse.sparray):
        array = scipy.sparse.load_npz(array)

    array = scipy.sparse.coo_array(array)
    return torch.sparse_coo_tensor(array.row, array.col, array.shape)  # type: ignore


def sparse_tensor_from_tensor(
    tensor: Union[torch.Tensor, PathLike, str],
) -> torch.Tensor:
    """Create a sparse tensor from a torch tensor or path to file.

    Caution: This function will load the tensor even if it is a dense tensor.
    Take care to ensure that the tensor is sufficiently sparse to avoid
    excessive memory usage.

    Args:
        tensor: Torch tensor or path to file.

    Returns:
        The original or loaded sparse tensor.

    Raises:
        ValueError: If the file is not a torch tensor or path to file.

    Returns:
        torch.Tensor: The sparse tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to_sparse()
    else:
        return torch.load(tensor, weights_only=True).to_sparse()


def load_tensor(
    tensor: Union[torch.Tensor, Iterable, PathLike, str],
) -> torch.Tensor:
    """Load a tensor from a file or tensor.

    Args:
        tensor: Tensor or path to file containing tensor.

    Returns:
        The original or loaded tensor.
    """
    try:
        loaded_tensor = tensor_from_tensor(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = tensor_from_numpy(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = tensor_from_df(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = torch.tensor(tensor)
    except Exception:
        raise ValueError(f"Failed to load tensor from {tensor}")

    return loaded_tensor


def load_sparse_tensor(
    tensor: Union[torch.Tensor, Iterable[Any], PathLike, str],
) -> torch.Tensor:
    """Load a sparse tensor from a file or tensor.

    Args:
        tensor (Union[torch.Tensor, PathLike]): Tensor or path to file
            containing tensor.

    Returns:
        torch.Tensor: The original or loaded tensor.
    """
    try:
        loaded_tensor = sparse_tensor_from_tensor(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = sparse_tensor_from_numpy(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = sparse_tensor_from_scipy(tensor)
    except Exception:
        pass
    try:
        loaded_tensor = sparse_tensor_from_df(tensor)  # type: ignore
    except Exception:
        pass
    try:
        loaded_tensor = torch.tensor(tensor)
    except Exception:
        raise ValueError(f"Failed to load tensor from {tensor}")

    assert loaded_tensor.layout == torch.sparse_coo

    return loaded_tensor


def load_array(
    array: Union[np.ndarray, Iterable, PathLike, str],
) -> np.ndarray:
    """Load a numpy array from an array, iterable, or file.

    Supported file formats:
    - npz
    - npy
    - csv
    - pt

    Args:
        array: numpy array, iterable, or path to file containing numpy array.

    Returns:
        The original or loaded numpy array.

    Raises:
        ValueError: If the array cannot be loaded from the given file or iterable.
    """
    try:
        loaded_array = array_from_numpy(array)  # type: ignore
    except Exception:
        pass
    try:
        loaded_array = array_from_tensor(array)  # type: ignore
    except Exception:
        pass
    try:
        loaded_array = array_from_df(array)  # type: ignore
    except Exception:
        pass
    try:
        loaded_array = np.array(array)
    except Exception:
        raise ValueError(f"Failed to load array from {array}")

    return loaded_array


def idx_1D_to_2D_tensor(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Convert 1D indices to 2D coordinates.

    Args:
        x (torch.Tensor): 1D indices tensor.
        m (int): Number of rows in the 2D grid.
        n (int): Number of columns in the 2D grid.

    Returns:
        torch.Tensor: 2D coordinates tensor of shape (len(x), 2).
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D_tensor(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """Convert 2D coordinates to 1D indices.

    Args:
        x (torch.Tensor): 2D coordinates tensor of shape (N, 2).
        m (int): Number of rows in the 2D grid.
        n (int): Number of columns in the 2D grid.

    Returns:
        torch.Tensor: 1D indices tensor.
    """
    return x[0] * n + x[1]


def print_cuda_mem_stats(device: Optional[torch.device] = None):
    """Print CUDA memory statistics for debugging."""
    f, t = torch.cuda.mem_get_info(device)
    print(f"Free/Total: {f / (1024**3):.2f}GB/{t / (1024**3):.2f}GB")


def count_parameters(model):
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
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


def profile_fn(
    fn,
    sort_by="cuda_time_total",
    row_limit=50,
    profile_kwargs={},
    fn_kwargs={},
):
    """Profile a function with PyTorch's profiler.

    Args:
        fn: Function to profile.
        sort_by (str, optional): Column to sort results by. Defaults to
            "cuda_time_total".
        row_limit (int, optional): Maximum number of rows to display.
            Defaults to 50.
        **fn_kwargs: Keyword arguments to pass to the function.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        **profile_kwargs,
    ) as prof:
        fn(**fn_kwargs)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def create_random_topographic_hh_connectivity(
    sheet_size: tuple[int, int],
    synapse_std: float,
    synapses_per_neuron: int,
    self_recurrence: bool,
) -> torch.Tensor:
    """Create random topographic hidden-to-hidden connectivity.

    Args:
        sheet_size (tuple[int, int]): Size of the sheet-like neural layer (rows,
            columns).
        synapse_std (float): Standard deviation of the Gaussian distribution for
            sampling synapse connections.
        synapses_per_neuron (int): Number of incoming synapses per neuron.
        self_recurrence (bool): Whether neurons can connect to themselves.

    Returns:
        torch.Tensor: Sparse connectivity matrix in COO format.
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
    input_indices: Optional[Union[torch.Tensor, PathLike]] = None,
) -> torch.Tensor:
    """Create identity connectivity for input-to-hidden connections.

    Args:
        input_size (int): Size of the input.
        num_neurons (int): Number of neurons in the hidden layer.
        input_indices (Union[torch.Tensor, PathLike], optional): Indices of
            neurons that receive input. If None, all neurons receive input from
            corresponding input indices. Defaults to None.

    Returns:
        torch.Tensor: Sparse connectivity matrix in COO format.

    Raises:
        ValueError: If input_indices are invalid.
    """
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
