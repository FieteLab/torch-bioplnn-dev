from collections.abc import Callable
from typing import Any, TypeVar, Union

import torch
from numpy.typing import NDArray

"""
Type alias for a function that initializes a tensor.

Should take a variable number of positional arguments describing the shape
of the tensor to initialize. Can optionally take a `device` keyword argument,
in which case the function is expected to return a tensor on that device.

Returns a torch.Tensor.
"""
TensorInitFnType = Callable[..., torch.Tensor]

"""
Type alias for a function that applies an activation to a tensor.

Takes a single torch.Tensor as input and returns a transformed torch.Tensor.
Used for neural network activation functions.
"""
ActivationFnType = Callable[[torch.Tensor], torch.Tensor]

T = TypeVar("T", bound=Any)

"""
Type alias for a 1D parameter.

Used to annotate function arguments that can be a single value, a list, or a 
NumPy array. Generic over the element type T.
"""
Param1dType = Union[T, list[T], NDArray[T]]

"""
Type alias for a 2D parameter.

Used to annotate function arguments that can be a single value, a nested list,
or a NumPy array. Generic over the element type T.
"""
Param2dType = Union[T, list[list[T]], NDArray[T]]
