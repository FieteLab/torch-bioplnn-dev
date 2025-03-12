from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

from bioplnn.models.connectome import ConnectomeODERNN, ConnectomeRNN
from bioplnn.models.ei_crnn import Conv2dEIRNN


class ConnectomeImageClassifier(nn.Module):
    def __init__(
        self,
        rnn_kwargs: Mapping[str, Any],
        num_classes: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = ConnectomeRNN(**rnn_kwargs)

        if self.rnn.output_indices is None:
            out_size = self.rnn.num_neurons
        else:
            out_size = len(self.rnn.output_indices)

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
        num_steps: Optional[int] = None,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outs, hs = self.rnn(x, num_steps=num_steps)

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
    def __init__(
        self,
        rnn_kwargs: Mapping[str, Any],
        num_classes: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.rnn = ConnectomeODERNN(**rnn_kwargs)

        if self.rnn.output_indices is None:
            out_size = self.rnn.num_neurons
        else:
            out_size = len(self.rnn.output_indices)

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
        num_steps: int,
        start_time: float = 0.0,
        end_time: float = 1.0,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        num_steps: Optional[int] = None,
        loss_all_timesteps: bool = False,
        return_activations: bool = False,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            list[torch.Tensor],
            list[list[torch.Tensor]],
            list[torch.Tensor],
        ]
    ):
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
