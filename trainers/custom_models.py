from math import prod

import torch
from torch import nn

from bioplnn.models import Conv2dEIRNN
from bioplnn.utils import get_activation_class


class AttentionalGainModulation(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_size: tuple[int, int]):
        """
        Initializes the SimpleAttentionalGain module.

        Args:
            spatial_size (tuple[int, int]): The spatial size of the input tensors (H, W).

        """
        super().__init__()
        self.spatial_size = spatial_size

        self.spatial_average = nn.AdaptiveAvgPool2d(spatial_size)
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bias = nn.Parameter(torch.zeros(1))  # init gain scaling to zero
        self.slope = nn.Parameter(torch.ones(1))  # init slope to one
        self.threshold = nn.Parameter(torch.zeros(1))  # init threshold to zero

        self.cached = None

    def forward(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
        op: str = "mul",
        use_cache: bool = False,
    ):
        """
        Forward pass of the EI model.

        Args:
            modulation (torch.Tensor): The modulation input.
            x (torch.Tensor): The x input.

        Returns:
            torch.Tensor: The output after applying gain modulation to the x input.
        """
        if use_cache:
            modulation = self.cached
        else:
            # Process modulation
            modulation = self.spatial_average(modulation)
            # Match channels
            modulation = self.match_channels(modulation)
            # Apply threshold shift
            modulation = modulation - self.threshold
            # Apply slope
            modulation = modulation * self.slope
            # Apply sigmoid & bias
            modulation = self.bias + (1 - self.bias) * torch.sigmoid(modulation)

        self.cached = modulation

        if op == "mul":
            return x * modulation
        else:
            raise NotImplementedError(f"Op {op} is not supported")


class LowRankModulation(nn.Module):
    def __init__(self, in_channels: int, spatial_size: tuple[int, int]):
        super().__init__()
        self.spatial_size = spatial_size

        self.spatial_average = nn.AdaptiveAvgPool2d((1, 1))
        self.rank_one_vec_h = nn.Linear(in_channels, spatial_size[0])
        self.rank_one_vec_w = nn.Linear(in_channels, spatial_size[1])

        self.cached = None

    def forward(
        self, x: torch.Tensor, modulation: list[torch.Tensor], op="add", use_cache=False
    ):
        # rank_one_vector = torch.matmul(input, self.W) + self.bias
        # # compute the rank one matrix
        # rank_one_perturbation = torch.matmul(rank_one_vector, rank_one_vector.transpose(-2, -1))
        # perturbed_input = input + rank_one_perturbation
        # return perturbed_input

        if use_cache:
            modulation = self.cached
        else:
            sp_vec = self.spatial_average(modulation)
            sp_vec = sp_vec.flatten(1)
            h_vec = self.rank_one_vec_h(sp_vec)
            w_vec = self.rank_one_vec_w(sp_vec)

            modulation = torch.bmm(h_vec.unsqueeze(-1), w_vec.unsqueeze(-2)).unsqueeze(
                -3
            )
            modulation = sp_vec.unsqueeze(-1).unsqueeze(-1) * modulation

        self.cached = modulation

        if op == "add":
            return x + modulation
        elif op == "mul":
            return x * torch.sigmoid(modulation)
        else:
            raise NotImplementedError(f"Op {op} is not supported")


class ConvModulation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: tuple[int, int] = (3, 3),
        activation: str = "relu",
        bias: bool = True,
    ):
        """
        Initializes the SimpleAttentionalGain module.

        Args:
            spatial_size (tuple[int, int]): The spatial size of the input tensors (H, W).

        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=bias,
        )
        self.activation = get_activation_class(activation)()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=bias,
        )
        self.cached = None

    def forward(
        self,
        x: torch.Tensor,
        modulation: list[torch.Tensor],
        op="mul",
        use_cache: bool = False,
    ):
        """
        Forward pass of the EI model.

        Args:
            modulation (torch.Tensor): The modulation input.
            x (torch.Tensor): The x input.

        Returns:
            torch.Tensor: The output after applying gain modulation to the x input.
        """

        if use_cache:
            modulation = self.cached
        else:
            modulation = self.activation(self.conv1(modulation))
            modulation = self.conv2(modulation)

        self.cached = modulation

        if op == "add":
            return x + modulation
        elif op == "mul":
            return x * torch.sigmoid(modulation)
        else:
            raise NotImplementedError(f"Op {op} is not supported")


class SelfAttnModulation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: tuple[int, int],
        kernel_size: tuple[int, int] = (3, 3),
        num_heads=8,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initializes the EI model.

        Args:
            in_channels (int): The number of input channels.
            spatial_size (tuple[int, int]): The spatial size of the input.

        """
        super().__init__()
        # Initialize the weight and bias matrices
        self.spatial_size = spatial_size
        embed_dim = spatial_size[0] * spatial_size[1]

        self.activation = get_activation_class(activation)()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Conv2d(
            out_channels,
            3 * out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=out_channels,
            bias=bias,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        conv1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=out_channels,
            bias=bias,
        )
        conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=out_channels,
            bias=bias,
        )
        self.ff = nn.Sequential(conv1, self.activation, conv2)
        self.cached = None

    def forward(
        self,
        x: torch.Tensor,
        modulation: list[torch.Tensor],
        op="mul",
        use_cache=False,
    ):
        """
        Forward pass of the model.

        Args:
            modulation (torch.Tensor): The modulation tensor.
            x (torch.Tensor): The x tensor.

        Returns:
            torch.Tensor: The output tensor after adding the rank one perturbation to the x.
        """
        # Compute the rank one matrix
        if use_cache:
            modulation = self.cached
        else:
            modulation = self.conv1(modulation)
            modulation_shape = modulation.shape
            modulation = self.norm1(modulation.flatten(2)).reshape(modulation_shape)
            q, k, v = self.qkv(modulation).flatten(2).chunk(3, dim=1)
            modulation = self.attn(q, k, v, need_weights=False)[0]
            modulation = self.norm2(modulation).reshape(modulation_shape)
            modulation = self.ff(modulation)

        self.cached = modulation

        if op == "add":
            return x + modulation
        elif op == "mul":
            return x * torch.sigmoid(modulation)
        else:
            raise NotImplementedError(f"Op {op} is not supported")


class ModulationWrapper(nn.Module):
    def __init__(
        self,
        modulations,
        modulation_modules,
        modulation_timestep,
        apply_timestep,
        num_steps,
        op,
        modulation_from_all_layers=False,
    ):
        super().__init__()
        self.modulation_modules = modulation_modules
        self.modulation_timestep = modulation_timestep
        self.apply_timestep = apply_timestep
        self.op = op

        num_layers = len(modulation_modules)

        for i in range(num_layers):
            if modulations[i].dim() == 4:
                if num_steps is None or num_steps < 1:
                    raise ValueError(
                        "If x is 4D, num_steps must be provided and greater than 0"
                    )
                modulations[i] = (
                    modulations[i].unsqueeze(0).expand((num_steps, -1, -1, -1, -1))
                )
            elif modulations[i].dim() == 5:
                if num_steps is not None and num_steps != modulations[i].shape[0]:
                    raise ValueError(
                        "If x is 5D and num_steps is provided, it must match the sequence length."
                    )
                num_steps = modulations[i].shape[0]
            else:
                raise ValueError(
                    "The input must be a 4D tensor or a 5D tensor with sequence length."
                )

        if modulation_from_all_layers:
            upsamplers = [
                nn.Upsample(size=modulation.shape[-2:]) for modulation in modulations
            ]
            modulations_new = [[] * num_steps for _ in range(num_layers)]
            for t in range(num_steps):
                upsampled_modulation = []
                for i in range(num_layers):
                    upsampled_modulation.append(upsamplers[i](modulations[i][t]))
                upsampled_modulation = torch.cat(upsampled_modulation, dim=1)
                for i in range(num_layers):
                    modulations_new[i][t] = upsampled_modulation

                if t == 0 and modulation_timestep != "all":
                    modulations_new = [
                        modulations_new[i][t]
                        .unsqueeze(0)
                        .expand((num_steps, -1, -1, -1, -1))
                        for i in range(num_layers)
                    ]
                    break
            modulations = modulations_new

        self.modulations = modulations

    def forward(self, x, i, t):
        if t in ("all", self.apply_timestep):
            if self.modulation_timestep == "same":
                modulation = self.modulations[i][t]
            else:
                modulation = self.modulations[self.modulation_timestep]

            use_cache = t > 0 and t != self.apply_timestep

            return self.modulation_modules(
                x,
                modulation,
                op=self.op,
                use_cache=use_cache,
            )
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        rnn_kwargs,
        modulation_enable=True,
        modulation_type: str = "ag",
        modulation_op: str = "mul",
        modulation_activation: str = "relu",
        modulation_num_heads: int = 8,
        modulation_dropout: float = 0.1,
        modulation_apply_to="hidden",
        modulation_timestep_cue="all",
        modulation_timestep_mix="all",
        modulation_from_all_layers=False,
        flush_hidden=True,
        flush_out=True,
        flush_fb=True,
        num_classes=6,
        fc_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        self.rnn = Conv2dEIRNN(batch_first=False, **rnn_kwargs)

        self.num_layers = rnn_kwargs["num_layers"]
        self.modulation_enable = modulation_enable
        self.modulation_type = modulation_type
        self.modulation_op = modulation_op
        self.modulation_activation = modulation_activation
        self.modulation_num_heads = modulation_num_heads
        self.modulation_dropout = modulation_dropout
        self.modulation_apply_to = modulation_apply_to
        self.modulation_timestep_cue = modulation_timestep_cue
        self.modulation_timestep_mix = modulation_timestep_mix
        self.modulation_from_all_layers = modulation_from_all_layers
        self.flush_hidden = flush_hidden
        self.flush_out = flush_out
        self.flush_fb = flush_fb

        if modulation_apply_to not in ("hidden", "layer_output"):
            raise ValueError("modulation_apply_to must be 'hidden' or 'layer_output'.")

        self.modulations = nn.ModuleList()
        self.modulations_inter = nn.ModuleList()

        if modulation_enable:
            for i in range(self.num_layers):
                if modulation_from_all_layers:
                    h_pyr_in_dim = sum(
                        [self.rnn.layers[j].h_pyr_dim for j in range(self.num_layers)]
                    )
                    h_inter_in_dim = sum(
                        [
                            self.rnn.layers[j].h_inter_dims_sum
                            for j in range(self.num_layers)
                        ]
                    )
                    out_in_dim = sum(
                        [self.rnn.layers[j].out_dim for j in range(self.num_layers)]
                    )
                else:
                    h_pyr_in_dim = self.rnn.layers[i].h_pyr_dim
                    h_inter_in_dim = self.rnn.layers[i].h_inter_dims_sum
                    out_in_dim = self.rnn.layers[i].out_dim

                if modulation_type == "ag":
                    modulation_class = AttentionalGainModulation
                    if modulation_apply_to == "hidden":
                        kwargs_pyr = {
                            "in_channels": h_pyr_in_dim,
                            "out_channels": self.rnn.layers[i].h_pyr_dim,
                            "spatial_size": self.rnn.layers[i].input_size,
                        }
                        kwargs_inter = {
                            "in_channels": h_inter_in_dim,
                            "out_channels": self.rnn.layers[i].h_inter_dims_sum,
                            "spatial_size": self.rnn.layers[i].inter_size,
                        }
                    else:
                        kwargs_out = {
                            "in_channels": out_in_dim,
                            "out_channels": self.rnn.layers[i].out_dim,
                            "spatial_size": self.rnn.layers[i].out_size,
                        }
                elif modulation_type == "lr":
                    modulation_class = LowRankModulation
                    if modulation_apply_to == "hidden":
                        kwargs_pyr = {
                            "in_channels": h_pyr_in_dim,
                            "spatial_size": self.rnn.layers[i].input_size,
                        }
                        kwargs_inter = {
                            "in_channels": h_inter_in_dim,
                            "spatial_size": self.rnn.layers[i].inter_size,
                        }
                    else:
                        kwargs_out = {
                            "in_channels": out_in_dim,
                            "spatial_size": self.rnn.layers[i].out_size,
                        }
                elif modulation_type == "self_attn":
                    modulation_class = SelfAttnModulation
                    kwargs_common = {
                        "kernel_size": self.rnn.exc_kernel_sizes[i],
                        "num_heads": modulation_num_heads,
                        "activation": modulation_activation,
                        "dropout": modulation_dropout,
                        "bias": self.rnn.biases[i],
                    }
                    if modulation_apply_to == "hidden":
                        kwargs_pyr = kwargs_common | {
                            "in_channels": h_pyr_in_dim,
                            "out_channels": self.rnn.layers[i].h_pyr_dim,
                            "spatial_size": self.rnn.layers[i].input_size,
                        }
                        kwargs_inter = kwargs_common | {
                            "in_channels": h_inter_in_dim,
                            "out_channels": self.rnn.layers[i].h_inter_dims_sum,
                            "spatial_size": self.rnn.layers[i].inter_size,
                        }
                    else:
                        kwargs_out = kwargs_common | {
                            "in_channels": out_in_dim,
                            "out_channels": self.rnn.layers[i].out_dim,
                            "spatial_size": self.rnn.layers[i].out_size,
                        }

                elif modulation_type == "conv":
                    modulation_class = ConvModulation
                    kwargs_common = {
                        "kernel_size": self.rnn.exc_kernel_sizes[i],
                        "activation": modulation_activation,
                        "bias": self.rnn.biases[i],
                    }
                    if modulation_apply_to == "hidden":
                        kwargs_pyr = kwargs_common | {
                            "in_channels": h_pyr_in_dim,
                            "out_channels": self.rnn.layers[i].h_pyr_dim,
                        }
                        kwargs_inter = kwargs_common | {
                            "in_channels": h_inter_in_dim,
                            "out_channels": self.rnn.layers[i].h_inter_dims_sum,
                        }
                    else:
                        kwargs_out = kwargs_common | {
                            "in_channels": out_in_dim,
                            "out_channels": self.rnn.layers[i].out_dim,
                        }
                else:
                    raise ValueError(
                        "modulation_type must be 'ag', 'lr', 'attn', or 'conv'"
                    )

                if modulation_apply_to == "hidden":
                    self.modulations.append(modulation_class(**kwargs_pyr))
                    self.modulations_inter.append(modulation_class(**kwargs_inter))
                else:
                    self.modulations.append(modulation_class(**kwargs_out))

        self.out_layer = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(
                self.rnn.layers[-1].out_dim * prod(self.rnn.layers[-1].out_size),
                fc_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def _format_modulation_timestep(
        self, modulation_timestep_cue, modulation_timestep_mix, num_steps
    ):
        modulation_timesteps = []
        for i, modulation_timestep in enumerate(
            (modulation_timestep_cue, modulation_timestep_mix)
        ):
            if modulation_timestep == "first":
                modulation_timestep = 0
            elif modulation_timestep == "last":
                modulation_timestep = num_steps - 1

            if (
                modulation_timestep not in ("all", "same")
                and -num_steps <= modulation_timestep < num_steps
            ):
                modulation_timestep = (num_steps + modulation_timestep) % num_steps
            elif (i == 0 and modulation_timestep != "same") or (
                i == 1 and modulation_timestep != "all"
            ):
                raise ValueError(
                    "modulation_timestep must be 'first', 'last', 'same'(for cue), 'all'(for mix)"
                    "or an integer in the range of (-num_steps, num_steps]."
                )
            modulation_timesteps.append(modulation_timestep)

        return modulation_timesteps

    def forward(
        self,
        cue: torch.Tensor,
        mix: torch.Tensor,
        num_steps: int = None,
        loss_all_timesteps: bool = False,
    ):
        outs, (h_pyrs, h_inters, fbs) = self.rnn(
            x=cue,
            num_steps=num_steps,
            return_all_layers_out=True,
        )

        h_pyr_0, h_inter_0, out_0, fb_0 = None, None, None, None
        if not self.flush_hidden:
            h_pyr_0 = [h[-1] for h in h_pyrs]
            h_inter_0 = [h[-1] for h in h_inters]
        if not self.flush_out:
            out_0 = [o[-1] for o in outs]
        if not self.flush_fb:
            fb_0 = [f[-1] for f in fbs]

        modulation_pyr_fn, modulation_inter_fn, modulation_out_fn = None, None, None
        if self.modulation_enable:
            num_steps = h_pyrs[0].shape[0]
            modulation_timestep_cue, modulation_timestep_mix = (
                self._format_modulation_timestep(
                    self.modulation_timestep_cue,
                    self.modulation_timestep_mix,
                    num_steps,
                )
            )
            if self.modulation_apply_to == "hidden":
                modulation_pyr_fn = ModulationWrapper(
                    modulations=h_pyrs,
                    modulation_modules=self.modulations,
                    modulation_timestep=modulation_timestep_cue,
                    apply_timestep=modulation_timestep_mix,
                    num_steps=num_steps,
                    op=self.modulation_op,
                    modulation_from_all_layers=self.modulation_from_all_layers,
                )

                modulation_inter_fn = ModulationWrapper(
                    modulations=h_inters,
                    modulation_modules=self.modulations_inter,
                    modulation_timestep=modulation_timestep_cue,
                    apply_timestep=modulation_timestep_mix,
                    num_steps=num_steps,
                    op=self.modulation_op,
                    modulation_from_all_layers=self.modulation_from_all_layers,
                )
            else:
                modulation_out_fn = ModulationWrapper(
                    modulations=outs,
                    modulation_modules=self.modulations,
                    modulation_timestep=modulation_timestep_cue,
                    apply_timestep=modulation_timestep_mix,
                    num_steps=num_steps,
                    op=self.modulation_op,
                    modulation_from_all_layers=self.modulation_from_all_layers,
                )

        outs, _ = self.rnn(
            x=mix,
            num_steps=num_steps,
            h_pyr_0=h_pyr_0,
            h_inter_0=h_inter_0,
            fb_0=fb_0,
            out_0=out_0,
            modulation_pyr_fn=modulation_pyr_fn,
            modulation_inter_fn=modulation_inter_fn,
            modulation_out_fn=modulation_out_fn,
            return_all_layers_out=False,
        )

        if loss_all_timesteps:
            return [self.out_layer(out.flatten(1)) for out in outs]

        return self.out_layer(outs[-1].flatten(1))
