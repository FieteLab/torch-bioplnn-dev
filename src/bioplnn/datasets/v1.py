import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

# def normalize_for_mp_np(indices, N_x=150, N_y=300, retina_radius=80):
#     x, y = indices
#     normalized_x = (1 - (x) / retina_radius) * 2.4 - 0.6
#     normalized_y = ((y - N_y // 2) / np.sqrt(retina_radius**2.0)) * 3.5
#     return normalized_x, normalized_y


# def r_theta_mp_np(data):
#     tmp = np.exp(data[0] + 1j * data[1]) - 0.5
#     return np.abs(tmp), np.angle(tmp)


# def image2v1(
#     image,
#     retina_indices,
#     image_top_corner=(4, 4),
#     N_x=150,
#     N_y=300,
#     retina_radius=80,
# ):
#     c, image_x, image_y = image.shape  # (C, H, W)
#     img_ind = np.zeros((2, image_x, image_y))
#     img_ind[0, :, :] = (
#         np.tile(0 + np.arange(image_x), (image_y, 1)).T / image_x * image_top_corner[0]
#     )
#     img_ind[1, :, :] = (
#         np.tile(np.arange(image_y) - image_y // 2, (image_x, 1))
#         / image_y
#         * image_top_corner[1]
#         * 2
#     )

#     flat_img_ind = img_ind.reshape((2, image_x * image_y))

#     normed_indices_retina = normalize_for_mp_np(retina_indices, N_x, N_y, retina_radius)
#     r_indices, theta_indices = r_theta_mp_np(normed_indices_retina)

#     v_field_x = r_indices * np.cos(theta_indices)
#     v_field_y = r_indices * np.sin(theta_indices)

#     device = image.device
#     image = image.cpu().numpy()

#     img_on_vfield = [
#         scipy.interpolate.griddata(
#             flat_img_ind.T,
#             im.flatten(),
#             np.array((v_field_x, v_field_y)).T,
#         )
#         for im in image
#     ]
#     img_on_vfield = np.stack(img_on_vfield)

#     img_on_vfield = torch.from_numpy(img_on_vfield).to(device).float()
#     img_on_vfield = torch.nan_to_num(img_on_vfield)

#     cortex = torch.zeros(c, 150, 300)
#     cortex[:, retina_indices[0], retina_indices[1]] = img_on_vfield

#     return cortex, v_field_x, v_field_y


def normalize_for_mp(indices, retina_radius, n):
    x, y = indices
    normed_x = (1 - x / retina_radius) * 2.4 - 0.6
    normed_y = ((y - n // 2) / retina_radius) * 3.5
    return normed_x, normed_y


def r_theta_mp(data):
    tmp = torch.exp(data[0] + 1j * data[1]) - 0.5
    return torch.abs(tmp), torch.angle(tmp)


class V1Dataset:
    def __init__(self, retina_path, m, n, retina_radius, image_top_corner=(4, 4)):
        """
        Read cortex information.
        """
        retina_indices = torch.load(retina_path)

        normed_indices = normalize_for_mp(retina_indices, retina_radius, n)
        r_indices, theta_indices = r_theta_mp(normed_indices)

        grid_x = r_indices * torch.cos(theta_indices)
        grid_y = r_indices * torch.sin(theta_indices)
        normed_grid_x = grid_x / (image_top_corner[0] / 2) - 1
        normed_grid_y = grid_y / (image_top_corner[1])

        self.grid = torch.full((1, m, n, 2), torch.nan)
        self.grid[:, retina_indices[0], retina_indices[1], :] = torch.stack(
            (normed_grid_y, normed_grid_x), dim=-1
        )

    def to_(self, device):
        self.grid = self.grid.to(device)

    def image_to_cortex(self, image: torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        grid = self.grid.expand(image.shape[0], -1, -1, -1)

        cortex = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            align_corners=True,
        )
        cortex = cortex.nan_to_num()
        return cortex.squeeze(0)


class MNIST_V1(MNIST, V1Dataset):
    def __init__(
        self,
        root,
        retina_path,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        m=150,
        n=300,
        retina_radius=80,
        image_top_corner=(4, 4),
    ):
        MNIST.__init__(
            self,
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        V1Dataset.__init__(
            self,
            retina_path=retina_path,
            m=m,
            n=n,
            retina_radius=retina_radius,
            image_top_corner=image_top_corner,
        )

    def __getitem__(self, index):
        image, target = MNIST.__getitem__(self, index)
        v1 = self.image_to_cortex(image).flatten(1).mean(0)
        return v1, target


class CIFAR10_V1(CIFAR10, V1Dataset):
    def __init__(
        self,
        root,
        retina_path,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        m=150,
        n=300,
        retina_radius=80,
        image_top_corner=(4, 4),
    ):
        CIFAR10.__init__(
            self,
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        V1Dataset.__init__(
            self,
            retina_path=retina_path,
            m=m,
            n=n,
            retina_radius=retina_radius,
            image_top_corner=image_top_corner,
        )

    def __getitem__(self, index):
        image, target = CIFAR10.__getitem__(self, index)
        v1 = self.image_to_cortex(image).flatten(1).mean(0)
        return v1, target


class CIFAR100_V1(CIFAR100, V1Dataset):
    def __init__(
        self,
        root,
        retina_path,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        m=150,
        n=300,
        retina_radius=80,
        image_top_corner=(4, 4),
    ):
        CIFAR100.__init__(
            self,
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        V1Dataset.__init__(
            self,
            retina_path=retina_path,
            m=m,
            n=n,
            retina_radius=retina_radius,
            image_top_corner=image_top_corner,
        )

    def __getitem__(self, index):
        image, target = CIFAR100.__getitem__(self, index)
        v1 = self.image_to_cortex(image).flatten(1).mean(0)
        return v1, target


if __name__ == "__main__":
    transform = T.ToTensor()
    dataset = CIFAR10_V1(
        "data",
        retina_path="connectivity/V1_indices.pt",
        transform=transform,
        train=True,
        download=True,
        m=150,
        n=300,
        retina_radius=80,
        image_top_corner=(4, 4),
    )
    image, target = dataset.__getitem__(1000)
    plt.imshow(image.T.reshape(150, 300, 3), cmap="gray")
    plt.show()
