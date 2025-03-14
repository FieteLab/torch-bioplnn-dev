from collections.abc import Mapping
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from bioplnn.models.connectome import ConnectomeODERNN, ConnectomeRNN
from bioplnn.models.ei_crnn import Conv2dEIRNN


class ConnectomeImageClassifier(nn.Module):
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

        if self.rnn.output_neurons is None:
            out_size = self.rnn.hidden_size
        else:
            out_size = len(self.rnn.output_neurons)

        self.out_layer = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(out_size, fc_dim),
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
        """Forward pass of the ConnectomeImageClassifier.

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


class ConnectomeODEImageClassifier(nn.Module):
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

        if self.rnn.output_neurons is None:
            out_size = self.rnn.num_neurons
        else:
            out_size = len(self.rnn.output_neurons)

        self.out_layer = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(out_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        num_steps: int,
        start_time: float = 0.0,
        end_time: float = 1.0,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the ConnectomeODEImageClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, ...].
            num_steps (int): Number of integration steps.
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
            start_time=start_time,
            end_time=end_time,
            num_steps=num_steps,
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


class CRNNImageClassifier(nn.Module):
    """Convolutional RNN-based image classifier.

    Uses a convolutional E/I RNN as the feature extractor followed by a linear
    classifier.

    Args:
        rnn_kwargs (Mapping[str, Any]): Keyword arguments to pass to the
            Conv2dEIRNN constructor.
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
        pool_size: tuple[int, int] = (1, 1),
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = Conv2dEIRNN(**rnn_kwargs)

        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.readout = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(
                self.rnn.layers[-1].out_channels * pool_size[0] * pool_size[1],
                fc_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

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
        """Forward pass of the CRNNImageClassifier.

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
        outs_last_layer = self.pool(outs_last_layer)

        if loss_all_timesteps:
            pred = torch.stack([self.readout(out) for out in outs_last_layer])
        else:
            pred = self.readout(outs_last_layer[-1])

        if return_activations:
            return pred, outs, h_neurons, fbs
        else:
            return pred
