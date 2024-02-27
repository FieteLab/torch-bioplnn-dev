import numpy as np
import scipy
import scipy.interpolate
import torch
from torch.profiler import ProfilerActivity, profile, record_function


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def idx_1D_to_2D(x, m, n):
    """
    Convert a 1D index to a 2D index.

    Args:
        x (torch.Tensor): 1D index.

    Returns:
        torch.Tensor: 2D index.
    """
    return torch.stack((x // m, x % n))


def idx_2D_to_1D(x, m, n):
    """
    Convert a 2D index to a 1D index.

    Args:
        x (torch.Tensor): 2D index.

    Returns:
        torch.Tensor: 1D index.
    """
    return x[0] * n + x[1]


def print_mem_stats():
    f, t = torch.cuda.mem_get_info()
    print(f"Free/Total: {f/(1024**3):.2f}GB/{t/(1024**3):.2f}GB")


def count_parameters(model):
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


def profile(fn, kwargs, sort_by="cuda_time_total", row_limit=50):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        fn(kwargs)
    return prof.key_averages.table(sort_by=sort_by, row_limit=row_limit)


def r_theta_mp(data):
    tmp = np.exp(data[0] + 1j * data[1]) - 0.5
    return np.abs(tmp), np.angle(tmp)


def normalize_for_mp(indices, N_x=150, N_y=300, retina_radius=80):
    x, y = indices
    normalized_x = (1 - (x) / retina_radius) * 2.4 - 0.6
    normalized_y = ((y - N_y // 2) / np.sqrt(retina_radius**2.0)) * 3.5
    return normalized_x, normalized_y


def flatten_indices(indices, N_y=300):
    return indices[0] * N_y + indices[1]


def image2v1(
    image,
    retina_indices,
    image_top_corner=(4, 4),
    N_x=150,
    N_y=300,
    retina_radius=80,
):
    image_x, image_y = image.shape[1:]  # (C, H, W)
    img_ind = np.zeros((2, image_x, image_y))
    img_ind[0, :, :] = (
        np.tile(0 + np.arange(image_x), (image_y, 1)).T
        / image_x
        * image_top_corner[0]
    )
    img_ind[1, :, :] = (
        np.tile(np.arange(image_y) - image_y // 2, (image_x, 1))
        / image_y
        * image_top_corner[1]
        * 2
    )

    flat_img_ind = img_ind.reshape((2, image_x * image_y))

    normed_indices_retina = normalize_for_mp(
        retina_indices, N_x, N_y, retina_radius
    )
    r_indices, theta_indices = r_theta_mp(normed_indices_retina)

    v_field_x = r_indices * np.cos(theta_indices)
    v_field_y = r_indices * np.sin(theta_indices)

    device = image.device
    image = image.cpu().numpy()

    if len(image.shape) == 3:
        img_on_vfield = [
            scipy.interpolate.griddata(
                flat_img_ind.T,
                im.flatten(),
                np.array((v_field_x, v_field_y)).T,
            )
            for im in image
        ]
        img_on_vfield = np.stack(img_on_vfield)
    else:
        img_on_vfield = scipy.interpolate.griddata(
            flat_img_ind.T,
            image[0].flatten(),
            np.array((v_field_x, v_field_y)).T,
        )

    img_on_vfield = torch.from_numpy(img_on_vfield).to(device).float()
    img_on_vfield = torch.nan_to_num(img_on_vfield)
    return img_on_vfield
