from collections.abc import Mapping
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from bioplnn.models.connectome import ConnectomeODERNN, ConnectomeRNN
from bioplnn.models.spatially_embedded import SpatiallyEmbeddedRNN

# TODO: Some docstrings may be outdated, might need to update


class ConnectomeClassifier(nn.Module):
    """Connectome-based image classifier.

    Uses a connectome RNN as the feature extractor followed by a linear
    classifier.

    Args:
        rnn_kwargs (Mapping[str, Any]): Keyword arguments to pass to the
            ConnectomeRNN constructor.
        num_classes (int): Number of output classes.
        fc_dim (int, optional): Dimension of the fully connected layer.
            Defaults to 512.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """

    def __init__(
        self,
        rnn_kwargs: Mapping[str, Any],
        num_classes: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = ConnectomeRNN(**rnn_kwargs)

        self.out_layer = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.rnn.output_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
        **rnn_forward_kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the ConnectomeClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, ...].
            loss_all_timesteps (bool, optional): If True, compute loss for all
                timesteps. Defaults to False.
            return_activations (bool, optional): If True, return activations.
                Defaults to False.
            **rnn_forward_kwargs: Additional keyword arguments to pass to the
                RNN forward method.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If return_activations is False, returns predictions tensor.
                If return_activations is True, returns a tuple of
                (predictions, outputs, hidden states).
        """
        # Hack for image inputs
        if x.ndim == 4:
            x = x.flatten(1)
        if x.ndim == 5:
            x = x.flatten(2)

        outs, hs = self.rnn(x, **rnn_forward_kwargs)

        if self.rnn.batch_first:
            outs = outs.transpose(0, 1)

        if loss_all_timesteps:
            pred = torch.stack([self.out_layer(out) for out in outs])
        else:
            pred = self.out_layer(outs[-1])

        if return_activations:
            return pred, outs, hs
        else:
            return pred


class ConnectomeODEClassifier(nn.Module):
    """Connectome ODE-based image classifier.

    Uses a connectome ODE RNN as the feature extractor followed by a linear
    classifier.

    Args:
        rnn_kwargs (Mapping[str, Any]): Keyword arguments to pass to the
            ConnectomeODERNN constructor.
        num_classes (int): Number of output classes.
        fc_dim (int, optional): Dimension of the fully connected layer.
            Defaults to 512.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """

    def __init__(
        self,
        rnn_kwargs: Mapping[str, Any],
        num_classes: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = ConnectomeODERNN(**rnn_kwargs)

        self.out_layer = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.rnn.output_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        num_evals: int,
        start_time: float = 0.0,
        end_time: float = 1.0,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the ConnectomeODEClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, ...].
            num_evals (int): Number of evaluations to return.
            start_time (float, optional): Start time for ODE integration.
                Defaults to 0.0.
            end_time (float, optional): End time for ODE integration.
                Defaults to 1.0.
            loss_all_timesteps (bool, optional): If True, compute loss for all
                timesteps. Defaults to False.
            return_activations (bool, optional): If True, return activations.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                If return_activations is False, returns predictions tensor.
                If return_activations is True, returns a tuple of
                (predictions, hidden states, timestamps).
        """
        # Hack for image inputs
        if x.ndim == 4:
            x = x.flatten(1)
        if x.ndim == 5:
            x = x.flatten(2)

        outs, hs, ts = self.rnn(
            x,
            num_evals=num_evals,
            start_time=start_time,
            end_time=end_time,
        )

        if self.rnn.batch_first:
            outs = outs.transpose(0, 1)

        if loss_all_timesteps:
            pred = torch.stack([self.out_layer(out) for out in outs])
        else:
            pred = self.out_layer(outs[-1])

        if return_activations:
            return pred, hs, ts
        else:
            return pred


class SpatiallyEmbeddedClassifier(nn.Module):
    """Spatially embedded RNN-based image classifier.

    Uses a convolutional E/I RNN as the feature extractor followed by a linear
    classifier.

    Args:
        rnn_kwargs (Mapping[str, Any]): Keyword arguments to pass to the
            SpatiallyEmbeddedRNN constructor.
        num_classes (int): Number of output classes.
        pool_size (tuple[int, int], optional): Spatial dimensions after pooling.
            Defaults to (1, 1).
        fc_dim (int, optional): Dimension of the fully connected layer.
            Defaults to 512.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """

    def __init__(
        self,
        rnn_kwargs: Mapping[str, Any],
        num_classes: int,
        pool_size_classifier: tuple[int, int] = (1, 1),
        pool_mode_classifier: str = "max",
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = SpatiallyEmbeddedRNN(**rnn_kwargs)

        if pool_mode_classifier == "avg":
            self.pool = nn.AdaptiveAvgPool2d(pool_size_classifier)
        elif pool_mode_classifier == "max":
            self.pool = nn.AdaptiveMaxPool2d(pool_size_classifier)
        else:
            raise ValueError(f"Invalid pool_mode: {pool_mode_classifier}")

        self.readout = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(
                self.rnn.areas[-1].out_channels
                * pool_size_classifier[0]
                * pool_size_classifier[1],  # type: ignore
                fc_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    @torch.compiler.disable(recursive=False)
    def forward(
        self,
        x: torch.Tensor,
        *,
        num_steps: Optional[int] = None,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            List[torch.Tensor],
            List[List[torch.Tensor]],
            List[torch.Tensor],
        ],
    ]:
        """Forward pass of the SpatiallyEmbeddedClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, channels, height, width].
            num_steps (int, optional): Number of RNN timesteps. None means use
                default from RNN. Defaults to None.
            loss_all_timesteps (bool, optional): If True, compute loss for all
                timesteps. Defaults to False.
            return_activations (bool, optional): If True, return activations.
                Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor],
                  List[List[torch.Tensor]], List[torch.Tensor]]]:
                If return_activations is False, returns predictions tensor.
                If return_activations is True, returns a tuple of
                (predictions, outputs, hidden neuron states, feedback signals).
        """
        outs, h_neurons, fbs = self.rnn(x, num_steps=num_steps)

        outs_last_layer = outs[-1]
        if self.rnn.batch_first:
            outs_last_layer = outs_last_layer.transpose(0, 1)

        if loss_all_timesteps:
            pred = torch.stack(
                [self.readout(self.pool(out)) for out in outs_last_layer]
            )
        else:
            pred = self.readout(self.pool(outs_last_layer[-1]))

        if return_activations:
            return pred, outs, h_neurons, fbs
        else:
            return pred
