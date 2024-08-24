from typing import Optional

import torch

from .ei_crnn import Conv2dEIRNN


class Conv2dEIRNNModulation(Conv2dEIRNN):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initialize the Conv2dEIRNN.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            h_pyr_dim (int | list[int]): Number of channels of the pyramidal neurons or a list of number of channels for each layer.
            h_inter_dims (list[int] | list[list[int]]): Number of channels of the interneurons or a list of number of channels for each layer.
            fb_dim (int | list[int]): Number of channels of the feedback activationsor a list of number of channels for each layer.
            exc_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for excitatory convolutions or a list of kernel sizes for each layer.
            inh_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for inhibitory convolutions or a list of kernel sizes for each layer.
            num_layers (int): Number of layers in the RNN.
            num_steps (int): Number of iterations to perform in each layer.
            num_classes (int): Number of output classes. If None, the activations of the final layer at the last time step will be output.
            fb_adjacency (Optional[torch.Tensor], optional): Adjacency matrix for feedback connections.
            pool_kernel_size (list[int, int] | list[list[int, int]], optional): Size of the kernel for pooling or a list of kernel sizes for each layer.
            pool_stride (list[int, int] | list[list[int, int]], optional): Stride of the pooling operation or a list of strides for each layer.
            bias (bool | list[bool], optional): Whether or not to add the bias or a list of booleans indicating whether to add bias for each layer.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported.
            fc_dim (int, optional): Dimension of the fully connected layer.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = None,
        h_pyr_0: Optional[torch.Tensor] = None,
        h_inter_0: Optional[torch.Tensor] = None,
        fb_0: Optional[torch.Tensor] = None,
        out_0: Optional[torch.Tensor] = None,
        modulation_op: Optional[str] = None,
        modulation_timestep: Optional[str | int] = None,
        return_all_layers_out: bool = False,
    ):
        """
        Performs forward pass of the Conv2dEIRNN.

        Args:

        Returns:
            torch.Tensor: Output tensor after pooling of shape (b, n), where n is the number of classes.
        """
        # Check if the input is consistent with the number of steps
        x, num_steps = self._format_x(x, num_steps)

        modulation_pyr, modulation_inter, modulation_out, modulation_timestep = (
            self._format_modulation(
                modulation_pyr,
                modulation_inter,
                modulation_out,
                modulation_timestep,
                num_steps,
            )
        )

        # Track current device and batch size
        device = x.device
        batch_size = x.shape[1]

        # Initialize hidden states
        h_pyrs, h_inters, fbs, outs = self._init_state(
            h_pyr_0, h_inter_0, fb_0, out_0, num_steps, batch_size, device=device
        )

        # Perform forward pass over num_steps
        for t in range(num_steps):
            for i, layer in enumerate(self.layers):
                use_cache = not (t == 0 or t == modulation_timestep)

                # Apply additive modulation
                if modulation_timestep in (t, "all") and modulation_op == "add":
                    h_pyrs[i][t] = self.modulations[i](
                        modulation_pyr[i][t], h_pyrs[i][t], use_cache=use_cache
                    )
                    h_inters[i][t] = self.modulations_inter[i](
                        modulation_inter[i][t], h_inters[i][t], use_cache=use_cache
                    )
                    outs[i][t] = self.modulations[i](
                        modulation_out[i][t], outs[i][t], use_cache=use_cache
                    )

                # Compute layer update and output
                h_pyrs[i][t], h_inters[i][t], outs[i][t] = layer(
                    input=(
                        x[t]
                        if i == 0
                        else (
                            outs[i - 1][t - 1]
                            if self.layer_time_delay
                            else outs[i - 1][t]
                        )
                    ),
                    h_pyr=h_pyrs[i][t - 1],
                    h_inter=h_inters[i][t - 1],
                    fb=fbs[i][t - 1] if self.receives_fb[i] else None,
                )

                # Apply multiplicative modulation
                if modulation_timestep in (t, "all") and self.modulation_op == "mul":
                    if self.modulation_apply_to == "hidden":
                        h_pyrs[i][t] = self.modulations[i](
                            modulation_pyr[i][t], h_pyrs[i][t], use_cache=use_cache
                        )
                        h_inters[i][t] = self.modulations_inter[i](
                            modulation_inter[i][t], h_inters[i][t], use_cache=use_cache
                        )
                    else:
                        outs[i][t] = self.modulations[i](
                            modulation_out[i][t], outs[i][t], use_cache=use_cache
                        )

                # Apply feedback
                if self.use_fb:
                    for j in self.fb_adjacency[i]:
                        fbs[j][t] += self.fb_convs[f"fb_conv_{i}_{j}"](outs[i][t])

        if not return_all_layers_out:
            outs = outs[-1]

        return outs, (h_pyrs, h_inters, fbs)
