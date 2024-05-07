import glob
import json
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from bioplnn.utils import compact, rescale, seed_worker


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


class CLEVRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        cues_path: str,
        max_num_images: Optional[int],
        clevr_transforms: Callable,
        max_n_objects: int = 10,
        split: str = "train",
        holdout: list = [],
        include_zero: bool = True,
        mode: str = "color",
    ):
        super().__init__()
        self.data_root = data_root
        self.clevr_transforms = clevr_transforms
        self.max_num_images = max_num_images
        self.data_path = os.path.join(data_root, "images", split)
        self.max_n_objects = max_n_objects
        self.split = split

        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"

        self.files, self.meta, self.meta_shapes = self.get_files()
        self.holdout = holdout
        self.include_zero = include_zero
        self.mode = mode

        print("*** Holding out: {}".format(self.holdout))
        print("*** Mode: {}".format(self.mode))

        self.color_dict = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "orange": (255, 69, 0),
            "gray": (128, 128, 128),
            "brown": (150, 75, 0),
            "teal": (0, 128, 128),
            "navy": (0, 0, 128),
            "maroon": (128, 0, 0),
            "olive": (128, 128, 0),
            "cyan": (0, 255, 255),
        }

        self.color_list = list(self.color_dict.keys())
        self.shape_list = ["cylinder", "cube", "sphere"]
        self.iter_thresh = 10

        cue_list = glob.glob(os.path.join(cues_path, "*.png"))
        self.cues = {k: {} for k in self.shape_list}

        for x in cue_list:
            shp, col = x.split("/")[-1].split(".")[0].split("_")
            if col == "orange":
                col = "shapecue"
            self.cues[shp][col] = Image.open(x).convert("RGB")

    def _get_color_cue(self, meta):
        # include condition where the label is 0
        tlen = len(meta)
        picklist = meta
        if np.random.rand() < 1.0 / tlen:
            picklist = np.setdiff1d(self.color_list, meta)

        idx = np.random.randint(len(picklist))
        col = picklist[idx]

        iter = 0
        while (col in self.holdout) and (iter < self.iter_thresh):
            idx = np.random.randint(len(picklist))
            col = picklist[idx]
            iter += 1

        # errored out
        if iter == self.iter_thresh:
            picklist = np.setdiff1d(self.color_list, self.holdout)
            idx = np.random.randint(len(picklist))
            col = picklist[idx]

        return col

    def _get_color_cue_holdout(self, meta):
        # include only from the holdouts
        tlen = len(self.holdout)
        val = int(np.random.rand() * tlen)
        col = self.holdout[val]
        return col

    def _get_shape_cue(self, meta):
        tlen = len(meta)
        picklist = meta
        idx = np.random.randint(len(picklist))
        shape = picklist[idx]
        return shape

    def gen_color_trial(self, img, meta):
        cue = img.copy()
        col = self._get_color_cue(meta)

        ###
        # if stage is valid OR eval and holdout wasn't empty
        ###
        if ((self.split == "val") or (self.split == "test")) and (
            len(self.holdout) != 0
        ):
            col = self._get_color_cue_holdout(meta)

        cue.paste(self.color_dict[col], [0, 0, cue.size[0], cue.size[1]])
        label = np.sum([x == col for x in meta])
        return cue, label

    def gen_shape_trial(self, meta_shapes):
        shape = self._get_shape_cue(meta_shapes)
        ###
        # primitive shape renderings
        ###
        """
        sz = 100
        if shape == 'cylinder':
            sz = (50, 100)
        cue = draw_shape((img.size[0], img.size[1]), shape, (img.size[0]/2, img.size[1]/2), size=sz, radius=100, color=(255, 255, 255))
        """
        cue = self.cues[shape]["shapecue"].copy()
        label = np.sum([x == shape for x in meta_shapes])
        return cue, label

    def gen_conjunction_trial(self, meta, meta_shapes):
        idx = np.random.randint(len(meta))
        col = meta[idx]
        shape = self._get_shape_cue(meta_shapes)
        ###
        # primitive shape renderings
        ###
        """
        sz = 100
        if shape == 'cylinder':
            sz = (50, 100)
        cue = draw_shape((img.size[0], img.size[1]), shape, (img.size[0]/2, img.size[1]/2), size=sz, radius=100, color=col)
        """
        cue = self.cues[shape][col].copy()
        label = np.logical_and(
            np.array([x == shape for x in meta_shapes]),
            np.array([x == col for x in meta]),
        ).sum()
        return cue, label

    def __getitem__(self, index: int):
        image_path = self.files[index]
        meta = self.meta[index]
        meta_shapes = self.meta_shapes[index]

        img = Image.open(image_path)
        img = img.convert("RGB")

        # create a cue and get the right numerical label
        if self.mode == "color":
            cue, label = self.gen_color_trial(img, meta)

        elif self.mode == "shape":
            cue, label = self.gen_shape_trial(meta_shapes)

        elif self.mode == "both":
            cue, label = self.gen_conjunction_trial(meta, meta_shapes)

        elif self.mode == "every":
            # _mode 0: only color cue
            # _mode 1: only shape cue
            # _mode 2: conjunction cue
            _mode = np.random.choice(3, 1)[0]
            if _mode == 0:
                cue, label = self.gen_color_trial(img, meta)
            elif _mode == 1:
                cue, label = self.gen_shape_trial(meta_shapes)
            elif _mode == 2:
                cue, label = self.gen_conjunction_trial(meta, meta_shapes)

        else:
            raise NotImplementedError

        return (self.clevr_transforms(cue), self.clevr_transforms(img), label)

    def __len__(self):
        return len(self.files)

    """Need to also include meta data to create "cues"
    """

    def get_files(self) -> list[str]:
        with open(
            os.path.join(self.data_root, f"scenes/CLEVR_{self.split}_scenes.json")
        ) as f:
            scene = json.load(f)
        paths: list[Optional[str]] = []
        meta: list[Optional[dict]] = []
        meta_shapes: list[Optional[dict]] = []

        total_num_images = len(scene["scenes"])
        i = 0

        while (
            self.max_num_images is None or len(paths) < self.max_num_images
        ) and i < total_num_images:
            num_objects_in_scene = len(scene["scenes"][i]["objects"])
            obj_colors = [x["color"] for x in scene["scenes"][i]["objects"]]
            obj_shapes = [x["shape"] for x in scene["scenes"][i]["objects"]]

            if num_objects_in_scene <= self.max_n_objects:
                image_path = os.path.join(
                    self.data_path, scene["scenes"][i]["image_filename"]
                )
                assert os.path.exists(image_path), f"{image_path} does not exist"
                paths.append(image_path)
                meta.append(obj_colors)
                meta_shapes.append(obj_shapes)
            i += 1
        spaths, smeta, smetashapes = (
            list(t)
            for t in zip(
                *sorted(zip(compact(paths), compact(meta), compact(meta_shapes)))
            )
        )

        return spaths, smeta, smetashapes


def get_dataloaders(
    data_root: str,
    cues_path: str,
    train_batch_size: int,
    val_batch_size: int,
    max_n_objects: int,
    resolution: tuple[int, int],
    num_train_images: Optional[int] = None,
    num_val_images: Optional[int] = None,
    holdout: list = [],
    mode: str = "color",
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
    train_dataset = CLEVRDataset(
        data_root=data_root,
        cues_path=cues_path,
        max_num_images=num_train_images,
        clevr_transforms=clevr_transforms,
        split="train",
        max_n_objects=max_n_objects,
        holdout=holdout,
        mode=mode,
    )
    val_dataset = CLEVRDataset(
        data_root=data_root,
        cues_path=cues_path,
        max_num_images=num_val_images,
        clevr_transforms=clevr_transforms,
        split="val",
        max_n_objects=max_n_objects,
        holdout=holdout,
        mode=mode,
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
