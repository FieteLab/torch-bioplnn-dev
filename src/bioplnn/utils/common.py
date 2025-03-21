from collections.abc import Iterable, Mapping
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

from bioplnn.typing import ScalarOrArray2dType, ScalarOrListLike, T


def pass_fn(*args, **kwargs):
    """A no-op function that accepts any arguments and does nothing.

    Args:
        *args: Any positional arguments.
        **kwargs: Any keyword arguments.
    """
    pass


def without_keys(d: Mapping, keys: list[str]) -> dict:
    """Creates a new dictionary without specified keys.

    Args:
        d (Mapping): Input dictionary.
        keys (list[str]): List of keys to exclude.

    Returns:
        dict: A new dictionary without the specified keys.
    """
    return {x: d[x] for x in d if x not in keys}


def is_list_like(x: Any) -> bool:
    """Determines if an object is list-like (iterable but not a string or mapping).

    Args:
        x (Any): Object to check.

    Returns:
        bool: True if the object is list-like, False otherwise.
    """
    if isinstance(x, (str, Mapping)):
        return False
    try:
        iter(x)
        if len(x) > 0:
            x[0]
    except Exception:
        return False
    return True


def dict_flatten(d, delimiter=".", key=None):
    """Flattens a nested dictionary into a single-level dictionary.

    Keys of the flattened dictionary will be the path to the value, with path
    components joined by delimiter.

    Args:
        d (dict): Dictionary to flatten.
        delimiter (str, optional): String to join key path components.
            Defaults to ".".
        key (str, optional): Current key prefix. Defaults to None.

    Returns:
        dict: Flattened dictionary.

    Raises:
        ValueError: If flattening would result in duplicate keys.
    """
    key = f"{key}{delimiter}" if key is not None else ""
    non_dicts = {
        f"{key}{k}": v for k, v in d.items() if not isinstance(v, dict)
    }
    dicts = {
        f"{key}{k}": v
        for _k, _v in d.items()
        if isinstance(_v, dict)
        for k, v in dict_flatten(_v, delimiter=delimiter, key=_k).items()
    }

    if in_both := dicts.keys() & non_dicts.keys():
        if len(in_both) > 1:
            raise ValueError(
                f"flattened keys {list(in_both)} used more than once in dict"
            )
        else:
            raise ValueError(
                f"flattened key {list(in_both)[0]} used more than once in dict"
            )

    return {**non_dicts, **dicts}


def expand_list(
    x: Optional[ScalarOrListLike[T]], n: int, depth: int = 0
) -> Union[list[T], NDArray[Any]]:
    """Expands a value to a list of length n.

    If x is already a list, then the list is returned unchanged.

    If x is not a list, then x is expanded to a list of length n.

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If T is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x (Any): The variable to expand.
        n (int): The number of lists or tuples to expand to.
        depth (int, optional): The depth x can be recursively indexed. A depth
            of -1 will assume x is of type list[T] and check if x is already of
            the correct length. Defaults to 0.
    Returns:
        list[Any]: Expanded list.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    inner = x
    try:
        for _ in range(depth + 1):
            if isinstance(inner, str):
                raise TypeError
            inner = inner[0]  # type: ignore
    except (IndexError, TypeError):
        return [x] * n  # type: ignore

    if x is None:
        assert depth == -1
        raise ValueError("x cannot be None if depth is -1.")

    if len(x) != n:  # type: ignore
        raise ValueError(f"x must have length {n}.")

    return x  # type: ignore


def expand_array_2d(
    x: Optional[ScalarOrArray2dType[T]], m: int, n: int, depth: int = 0
) -> NDArray[Any]:
    """Expands a value to a 2D numpy array of shape (m, n).

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If x is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x (Any): The variable to expand.
        m (int): The number of rows in the expanded array.
        n (int): The number of columns in the expanded array.
        depth (int, optional): The depth x can be recursively indexed. A depth
            of -1 will assume x is of type list[T] and check if x is already of
            the correct shape.

    Returns:
        np.ndarray: Expanded 2D numpy array.
    """

    if m < 1 or n < 1:
        raise ValueError("m and n must be at least 1.")

    inner = x
    try:
        for _ in range(depth + 2):
            inner = inner[0]  # type: ignore
    except TypeError:
        array = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                array[i, j] = x
        return array

    if x is None:
        assert depth == -1
        raise ValueError("x cannot be None if depth is -1.")

    x = np.array(x, dtype=object)

    if x.shape != (m, n):
        raise ValueError(f"x must have shape ({m}, {n}).")

    return x


def check_possible_values(
    param_name: str,
    params: Union[Iterable, NDArray[Any], torch.Tensor],
    valid_values: Union[Iterable, NDArray[Any], torch.Tensor],
) -> None:
    """Check if the provided parameters are all valid values.

    Args:
        param_name (str): The name of the parameter (for error message).
        params (Iterable): The parameters to check.
        valid_values (Iterable): The valid values to check against.

    Raises:
        ValueError: If any of the parameters are not one of the valid values.
    """
    if isinstance(params, torch.Tensor):
        params = set(torch.unique(params).tolist())
    elif isinstance(params, np.ndarray):
        params = set(np.unique(params).tolist())
    else:
        params = set(params)

    if isinstance(valid_values, torch.Tensor):
        valid_values = set(torch.unique(valid_values).tolist())
    elif isinstance(valid_values, np.ndarray):
        valid_values = set(np.unique(valid_values).tolist())
    else:
        valid_values = set(valid_values)

    if not params <= valid_values:
        raise ValueError(f"{param_name} must be one of {valid_values}.")
