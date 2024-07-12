import glob
import json
import multiprocessing
import os
from typing import Callable, Optional

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from bioplnn.utils import rescale, seed_worker


def draw_shape(
    image_size, shape_type, center, size=None, radius=None, color=(255, 0, 0)
):
    # Create a new image with a white background
    image = Image.new("RGB", image_size, "gray")
    draw = ImageDraw.Draw(image)

    if shape_type == "cylinder":
        width, height = size[0], size[1]

        # Draw top cap (ellipse)
        x1_top, y1_top = center[0] - size[0] // 2, center[1] + size[1] // 2 - 25
        x2_top, y2_top = center[0] + size[0] // 2, center[1] + size[1] // 2 + 25
        draw.ellipse([x1_top, y1_top, x2_top, y2_top], fill=color)

        # Draw body (rectangle)
        x1_body, y1_body = center[0] - size[0] // 2, center[1] - size[1] // 2
        x2_body, y2_body = center[0] + size[0] // 2, center[1] + size[1] // 2
        draw.rectangle([x1_body, y1_body, x2_body, y2_body], fill=color)

        # Draw bottom cap (ellipse)
        x1_bottom, y1_bottom = center[0] - size[0] // 2, center[1] - size[1] // 2 - 25
        x2_bottom, y2_bottom = center[0] + size[0] // 2, center[1] - size[1] // 2 + 25
        draw.ellipse([x1_bottom, y1_bottom, x2_bottom, y2_bottom], fill=color)

    elif shape_type == "cube":
        if size is None:
            raise ValueError("Size must be provided for a square.")
        x1, y1 = center[0] - size // 2, center[1] - size // 2
        x2, y2 = center[0] + size // 2, center[1] + size // 2
        draw.rectangle([x1, y1, x2, y2], fill=color)

    elif shape_type == "sphere":
        if radius is None:
            raise ValueError("Radius must be provided for a circle.")
        x1, y1 = center[0] - radius, center[1] - radius
        x2, y2 = center[0] + radius, center[1] + radius
        draw.ellipse([x1, y1, x2, y2], fill=color)

    else:
        raise ValueError("Invalid shape_type. Use 'cylinder', 'square', or 'circle'.")

    return image


class qCLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        assets_path: str,
        clevr_transforms: Callable,
        return_images: bool = False,
        split: str = "train",
        holdout: list = [],
        mode: str = "color",
        primitive: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.return_images = return_images
        self.mode = mode
        self.holdout = holdout
        self.split = split
        self.primitive = primitive
        self.num_workers = num_workers

        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "valid" or self.split == "test"

        self._modes = (
            ["color", "shape", "conjunction"] if self.mode == "every" else [self.mode]
        )
        self.data_paths = {
            _mode: os.path.join(data_root, "{}_{}".format(split, _mode), "images")
            for _mode in self._modes
        }
        assert all(
            [os.path.exists(data_path) for data_path in self.data_paths.values()]
        )

        print("*** Holding out: {}".format(self.holdout))
        print("*** Mode: {}".format(self.mode))

        # populate this by reading in meta data
        self.files, self.cues, self.counts, self.modes = self.get_files()

        assert (
            len(self.files) != 0
        ), "Something about the config results in an empty dataset!"

        """object colors: gray, red, blue, green, brown, purple, cyan, yellow
        aux colors: black, white, pink, orange, teal, navy, maroon, olive
        """
        self.color_dict = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (173, 35, 35),
            "green": (29, 105, 20),
            "blue": (42, 75, 215),
            "yellow": (255, 238, 51),
            "purple": (129, 38, 192),
            "pink": (255, 192, 203),
            "orange": (255, 69, 0),
            "gray": (87, 87, 87),
            "brown": (129, 74, 25),
            "teal": (0, 128, 128),
            "navy": (0, 0, 128),
            "maroon": (128, 0, 0),
            "olive": (128, 128, 0),
            "cyan": (41, 208, 208),
        }

        self.color_list = list(self.color_dict.keys())
        self.shape_list = ["cylinder", "cube", "sphere"]
        self.iter_thresh = 10

        asset_list = glob.glob(os.path.join(assets_path, "*.png"))
        self.assets = {k: {} for k in self.shape_list}

        """populate the shape cue images
        """
        for x in asset_list:
            shp, col = x.split("/")[-1].split(".")[0].split("_")
            if col == "orange":
                col = "shapecue"
            self.assets[shp][col] = Image.open(x).convert("RGB")

    def gen_shape(self, img, shape, sz=100):
        if self.primitive:
            if shape == "cylinder":
                sz = (50, 100)
            cue = draw_shape(
                (img.size[0], img.size[1]),
                shape,
                (img.size[0] / 2, img.size[1] / 2),
                size=sz,
                radius=100,
                color=(255, 255, 255),
            )
        else:
            cue = self.assets[shape]["shapecue"].copy()
        return cue

    def gen_conjunction_trial(self, img, cue_str):
        col = cue_str[1]
        shape = cue_str[0]
        if self.primitive:
            sz = 100
            if shape == "cylinder":
                sz = (50, 100)
            cue = draw_shape(
                (img.size[0], img.size[1]),
                shape,
                (img.size[0] / 2, img.size[1] / 2),
                size=sz,
                radius=100,
                color=col,
            )
        else:
            cue = self.assets[shape][col].copy()

        return cue

    def get_file(self, _mode, scene_path):
        with open(scene_path, "r") as f:
            x = json.load(f)
            cue_type = x["cue"]
            if _mode == "conjunction":
                cue_type = "{}_{}".format(cue_type[0], cue_type[1])
            if (self.split == "train" and cue_type not in self.holdout) or (
                self.split in ("valid", "test")
                and (len(self.holdout) == 0 or cue_type in self.holdout)
            ):
                image_path = os.path.join(self.data_paths[_mode], x["image_filename"])
                assert os.path.exists(image_path), f"{image_path} does not exist"
                return image_path, x["cue"], x["target_count"], _mode
            return None, None, None, None

    def get_files(self) -> list[str]:
        paths = []
        cues = []
        counts = []
        modes = []
        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        for _mode in self._modes:
            spath = os.path.join(self.data_root, f"{self.split}_{_mode}", "scenes")
            scene_paths = sorted(glob.glob(os.path.join(spath, "*")))
            if self.num_workers > 1:
                path, cue, count, mode = zip(
                    *pool.starmap(
                        self.get_file, zip([_mode] * len(scene_paths), scene_paths)
                    )
                )
            else:
                path, cue, count, mode = zip(
                    *[self.get_file(_mode, x) for x in scene_paths]
                )
            path = filter(lambda x: x is not None, path)
            cue = filter(lambda x: x is not None, cue)
            count = filter(lambda x: x is not None, count)
            mode = filter(lambda x: x is not None, mode)
            paths.extend(path)
            cues.extend(cue)
            counts.extend(count)
            modes.extend(mode)

        return paths, cues, counts, modes

    # def visualize(self, output_dict):

    def __getitem__(self, index: int):
        image_path = self.files[index]
        cue_str = self.cues[index]
        label = self.counts[index]
        mode = self.modes[index]

        img = Image.open(image_path)
        img = img.convert("RGB")

        # create a cue and get the right numerical label
        if mode == "color":
            cue = img.copy()
            cue.paste(self.color_dict[cue_str], [0, 0, cue.size[0], cue.size[1]])
        elif mode == "shape":
            cue = self.gen_shape(img, cue_str)
        elif mode == "conjunction":
            cue = self.gen_conjunction_trial(img, cue_str)
        else:
            raise NotImplementedError

        if self.return_images:
            if mode == "conjunction":
                cue_str = "{}_{}".format(cue_str[0], cue_str[1])
            return (
                self.clevr_transforms(cue),
                self.clevr_transforms(img),
                label,
                image_path,
                mode,
                cue_str,
            )
        else:
            return self.clevr_transforms(cue), self.clevr_transforms(img), label

    def __len__(self):
        return len(self.files)


def get_qclevr_dataloaders(
    data_root: str,
    assets_path: str,
    train_batch_size: int,
    val_batch_size: int,
    resolution: tuple[int, int],
    holdout: list = [],
    mode: str = "color",
    primitive: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
):
    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),
            transforms.Resize(resolution),
        ]
    )
    train_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="train",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
    )
    val_dataset = qCLEVRDataset(
        data_root=data_root,
        assets_path=assets_path,
        clevr_transforms=clevr_transforms,
        split="valid",
        holdout=holdout,
        mode=mode,
        primitive=primitive,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    return train_dataloader, val_dataloader
