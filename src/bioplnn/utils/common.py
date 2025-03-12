from collections.abc import Mapping
from typing import Any

import numpy as np
from addict import Dict


class AttrDict(Dict):
    def __missing__(self, key: Any):
        raise KeyError(key)


def pass_fn(*args, **kwargs):
    pass


def without_keys(d: Mapping, keys: list[str]) -> dict:
    return {x: d[x] for x in d if x not in keys}


def is_list_like(x: Any) -> bool:
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

    if in_both := set(dicts.keys()) & set(non_dicts.keys()):
        if len(in_both) > 1:
            raise ValueError(
                f"flattened keys {list(in_both)} used more than once in dict"
            )
        else:
            raise ValueError(
                f"Key {in_both.pop()} used more than once in dict"
            )
    return non_dicts | dicts


def expand_list(x: Any, n: int, depth: int = 0) -> list[Any]:
    """
    Expand a variable x of type T to a list of x of length n, iff the variable
    is not already an iterable of T of length n.

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If T is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x: The variable to expand.
        n: The number of lists or tuples to expand to.
        depth: The depth x can be recursively indexed. A depth of -1 will
            assume x is of type list[T] and check if x is already of the
            correct length.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    inner = x
    try:
        for _ in range(depth + 1):
            if isinstance(inner, str):
                raise TypeError
            inner = inner[0]  # type: ignore
    except TypeError:
        return [x] * n  # type: ignore

    if x is None:
        assert depth == -1
        raise ValueError("x cannot be None if depth is -1.")

    if len(x) != n:
        raise ValueError(f"x must have length {n}.")

    return x


def expand_array_2d(x: Any, m: int, n: int, depth: int = 0) -> np.ndarray:
    """
    Expand a variable x of type T to a 2D array of shape (m, n) if the
    variable is not already an array of T of shape (m, n).

    Use depth > 0 if the intended type T can be indexed recursively, where
    depth is the maximum number of times x can be recursively indexed if of
    type T. For example, if x is a shallow list, then depth = 1. If x is a
    list of lists or an array or tensor, then depth = 2.

    Args:
        x: The variable to expand.
        m: The number of rows in the expanded array.
        n: The number of columns in the expanded array.
        depth: The depth x can be recursively indexed. A depth of -1 will
            assume x is of type list[T] and check if x is already of the
            correct shape.
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
