import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CABCDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.transform = transform
        self.metadata = np.load(os.path.join(self.root, "metadata", "combined.npy"))

        if train:
            self.metadata = self.metadata[: int(len(self.metadata) * 0.8)]
        else:
            self.metadata = self.metadata[int(len(self.metadata) * 0.8) :]

        self.image_dirs = self.metadata[:, 0].astype(str)
        self.image_names = self.metadata[:, 2].astype(str)
        self.labels = self.metadata[:, 4].astype(int)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_dirs[idx], self.image_names[idx])
        image = Image.open(img_path)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
