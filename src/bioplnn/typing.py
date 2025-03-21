from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any, TypeVar, Union

import torch
from numpy.typing import NDArray

TensorInitFnType = Callable[..., torch.Tensor]
"""Type alias for a function that initializes a tensor.

A function that takes a variable number of positional arguments describing the
shape of the tensor to initialize. Can optionally take a `device` keyword
argument, in which case the function is expected to return a tensor on that
device.

Returns:
    torch.Tensor: The initialized tensor.
"""
ActivationFnType = Callable[[torch.Tensor], torch.Tensor]
"""Type alias for a function that applies an activation to a tensor.

A function that takes a single torch.Tensor as input and returns a transformed
torch.Tensor of the same shape. Used for neural network activation functions.

Args:
    tensor: Input tensor to be transformed.

Returns:
    torch.Tensor: The transformed tensor.
"""

T = TypeVar("T")

PathLikeType = Union[PathLike, str]
"""Type alias for a path-like object or string.

Used to annotate function arguments that can be a path-like object or string.
"""

ListLike = Union[Sequence[T], NDArray[Any]]
"""Type alias for a sequence of values.

Used to annotate function arguments that can be a Sequence or 1D NumPy array.
Generic over the element type T.
"""

ScalarOrListLike = Union[T, ListLike[T]]
"""Type alias for a single value or a list-like of values.

Used to annotate function arguments that can be a single value, a Sequence, or a
1D NumPy array. Generic over the element type T.
"""

NeuronTypeParam = ScalarOrListLike[T]
"""Type alias for `ScalarOrListLike`.

Used to annotate function arguments and class attributes that, if single-valued,
apply to all neuron types, and if list-like, apply to each neuron type
(length must match the number of neuron types).
"""

Array2dType = Union[Sequence[Sequence[T]], NDArray[Any]]
"""Type alias for a 2D parameter.

Used to annotate function arguments that can be a nested list, or a 2D NumPy array.
Generic over the element type T.
"""

ScalarOrArray2dType = Union[T, Array2dType[T]]
"""Type alias for a single value or a 2D parameter.

Used to annotate function arguments that can be a single value, a nested Sequence,
or a 2D NumPy array. Generic over the element type T.
"""

CircuitParam = ScalarOrArray2dType[T]
"""Type alias for `ScalarOrArray2dType`.

Used to annotate function arguments and class attributes that, if single-valued,
apply to all connections in an area's circuit motif, and if 2D-array-like, apply to
each connection in the circuit motif (shape must match that of the connectivity
matrix for the circuit motif).
"""

InterAreaParam = ScalarOrArray2dType[T]
"""Type alias for `ScalarOrArray2dType`.

Used to annotate function arguments and class attributes that, if single-valued,
apply to all connections between areas, and if 2D-array-like, apply to each
connection between areas (shape must match that of the connectivity matrix between
areas).
"""
