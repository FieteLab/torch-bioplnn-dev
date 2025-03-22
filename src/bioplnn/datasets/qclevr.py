import glob
import json
import multiprocessing as mp
import os
from typing import Callable, Optional

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

COLORS = [
    "blue",
    "green",
    "gray",
    "red",
    "brown",
    "purple",
    "cyan",
    "yellow",
    "black",
    "white",
    "pink",
    "orange",
    "teal",
    "navy",
    "maroon",
    "olive",
]
SHAPES = ["cube", "cylinder", "sphere"]
CONJUNCTIONS = [
    "cube_blue",
    "cube_green",
    "cube_gray",
    "cube_red",
    "cube_brown",
    "cube_purple",
    "cube_cyan",
    "cube_yellow",
    "cube_black",
    "cube_white",
    "cube_pink",
    "cube_orange",
    "cube_teal",
    "cube_navy",
    "cube_maroon",
    "cube_olive",
    "sphere_gray",
    "sphere_red",
    "sphere_blue",
    "sphere_green",
    "sphere_brown",
    "sphere_purple",
    "sphere_cyan",
    "sphere_yellow",
    "sphere_black",
    "sphere_white",
    "sphere_pink",
    "sphere_orange",
    "sphere_teal",
    "sphere_navy",
    "sphere_maroon",
    "sphere_olive",
    "cylinder_gray",
    "cylinder_red",
    "cylinder_blue",
    "cylinder_green",
    "cylinder_brown",
    "cylinder_purple",
    "cylinder_cyan",
    "cylinder_yellow",
    "cylinder_black",
    "cylinder_white",
    "cylinder_pink",
    "cylinder_orange",
    "cylinder_teal",
    "cylinder_navy",
    "cylinder_maroon",
    "cylinder_olive",
]


class QCLEVRDataset(Dataset):
    def __init__(
        self,
        root: str,
        cue_assets_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
        split: str = "train",
        mode: str = "color",
        holdout: list = [],
        primitive: bool = True,
        shape_cue_color: str = "orange",
        use_cache: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.root = root
        self.transform = transform if transform is not None else lambda x: x
        self.return_metadata = return_metadata
        self.split = "valid" if split == "val" else split
        self.mode = mode
        self.holdout = sorted(holdout)
        self.primitive = primitive
        self.shape_cue_color = shape_cue_color
        self.num_workers = num_workers

        if num_workers > 1 and not use_cache:
            mp.set_start_method("fork", force=True)

        if root.endswith("/"):
            root = root[:-1]
        if not os.path.exists(root):
            raise ValueError(f"root path '{root}' does not exist")
        if not (
            self.split == "train"
            or self.split == "valid"
            or self.split == "test"
        ):
            raise ValueError("split must be 'train', 'valid', or 'test'")

        self.modes_avail = (
            ["color", "shape", "conjunction"]
            if self.mode == "every"
            else [self.mode]
        )
        self.data_paths = {
            _mode: os.path.join(
                root, "{}_{}".format(self.split, _mode), "images"
            )
            for _mode in self.modes_avail
        }
        if not all(
            [
                os.path.exists(data_path)
                for data_path in self.data_paths.values()
            ]
        ):
            raise ValueError("Data paths for all modes must exist.")

        print("*** Holding out: {}".format(self.holdout))
        print("*** Mode: {}".format(self.mode))

        # populate this by reading in meta data
        parent_dir = os.path.dirname(self.root)
        cache_dir = os.path.join(
            parent_dir,
            "qclevr_cache",
            self.split,
            mode,
            f"holdout={self.holdout}",
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "cache.json")
        if use_cache:
            cache = json.load(open(cache_path, "r"))
            self.files = cache["files"]
            self.cues = cache["cues"]
            self.counts = cache["counts"]
            self.modes = cache["modes"]
        else:
            self.files, self.cues, self.counts, self.modes = self.get_files()
            json.dump(
                {
                    "files": self.files,
                    "cues": self.cues,
                    "counts": self.counts,
                    "modes": self.modes,
                },
                open(cache_path, "w"),
            )

        if len(self.files) == 0:
            raise ValueError(
                "No files found in the dataset. Check your parameters."
            )

        """
        object colors: gray, red, blue, green, brown, purple, cyan, yellow
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

        if not primitive:
            if cue_assets_root is None:
                raise ValueError(
                    "cue_assets_root must be provided if primitive is False"
                )
            cue_paths = glob.glob(os.path.join(cue_assets_root, "*.png"))
            self.cue_assets = {k: {} for k in ["cylinder", "cube", "sphere"]}
            for path in cue_paths:
                shape, color = path.split("/")[-1].split(".")[0].split("_")
                self.cue_assets[shape][color] = Image.open(path).convert("RGB")
        else:
            if cue_assets_root is not None:
                raise ValueError(
                    "cue_assets_root must be None if primitive is True"
                )

    def get_file(self, mode, scene_path):
        scene = json.load(open(scene_path, "r"))
        cue_type = scene["cue"]
        if mode == "conjunction":
            cue_type = f"{cue_type[0]}_{cue_type[1]}"
        if (self.split == "train" and cue_type not in self.holdout) or (
            self.split in ("valid", "test")
            and (len(self.holdout) == 0 or cue_type in self.holdout)
        ):
            image_path = os.path.join(
                self.data_paths[mode], scene["image_filename"]
            )
            if not os.path.exists(image_path):
                raise ValueError(f"Image path '{image_path}' does not exist")
            return image_path, scene["cue"], scene["target_count"], mode
        return None, None, None, None

    def get_files(self):
        paths = []
        cues = []
        counts = []
        modes = []
        if self.num_workers > 1:
            pool = mp.Pool(self.num_workers)
        for _mode in self.modes_avail:
            spath = os.path.join(self.root, f"{self.split}_{_mode}", "scenes")
            scene_paths = sorted(glob.glob(os.path.join(spath, "*")))
            if self.num_workers > 1:
                path, cue, count, mode = zip(
                    *pool.starmap(
                        self.get_file,
                        zip([_mode] * len(scene_paths), scene_paths),
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

        if not (len(paths) == len(cues) == len(counts) == len(modes)):
            raise ValueError(
                "Length of paths, cues, counts, and modes must be equal, "
                f"currently {len(paths)}, {len(cues)}, {len(counts)}, {len(modes)}"
            )

        return paths, cues, counts, modes

    @staticmethod
    def draw_shape(
        image_size, shape_type, size=None, radius=None, color=(255, 0, 0)
    ):
        # Create a new image with a black background
        image = Image.new("RGB", image_size, "black")
        draw = ImageDraw.Draw(image)

        scale_factor = 1 - torch.rand((1,)).item() / 4.0

        # Determine random center coordinates based on the shape type
        if shape_type == "cylinder":
            if size is None:
                raise ValueError("Size must be provided for a cylinder.")
            width, height = (
                int(size[0] * scale_factor),
                int(size[1] * scale_factor),
            )
            cap_size = 25

            # Calculate the maximum allowed coordinates for the shape to be fully inside the canvas
            max_x = image_size[0] - width // 2 - cap_size
            max_y = image_size[1] - height // 2 - cap_size
            min_x = width // 2 + cap_size
            min_y = height // 2 + cap_size

            # Randomize the center within these bounds
            center = (
                torch.randint(min_x, max_x, (1,)).item(),
                torch.randint(min_y, max_y, (1,)).item(),
            )

            # Draw top cap (ellipse)
            x1_top, y1_top = (
                center[0] - width // 2,
                center[1] + height // 2 - cap_size,
            )
            x2_top, y2_top = (
                center[0] + width // 2,
                center[1] + height // 2 + cap_size,
            )
            draw.ellipse([x1_top, y1_top, x2_top, y2_top], fill=color)

            # Draw body (rectangle)
            x1_body, y1_body = center[0] - width // 2, center[1] - height // 2
            x2_body, y2_body = center[0] + width // 2, center[1] + height // 2
            draw.rectangle([x1_body, y1_body, x2_body, y2_body], fill=color)

            # Draw bottom cap (ellipse)
            x1_bottom, y1_bottom = (
                center[0] - width // 2,
                center[1] - height // 2 - cap_size,
            )
            x2_bottom, y2_bottom = (
                center[0] + width // 2,
                center[1] - height // 2 + cap_size,
            )
            draw.ellipse(
                [x1_bottom, y1_bottom, x2_bottom, y2_bottom], fill=color
            )

        elif shape_type == "cube":
            if size is None:
                raise ValueError("Size must be provided for a cube.")

            # Calculate the maximum allowed coordinates for the shape to be fully inside the canvas
            max_x = image_size[0] - size // 2 - 1
            max_y = image_size[1] - size // 2 - 1
            min_x = size // 2 + 1
            min_y = size // 2 + 1

            # Randomize the center within these bounds
            center = (
                torch.randint(min_x, max_x, (1,)).item(),
                torch.randint(min_y, max_y, (1,)).item(),
            )

            size = size * scale_factor
            x1, y1 = center[0] - size // 2, center[1] - size // 2
            x2, y2 = center[0] + size // 2, center[1] + size // 2
            draw.rectangle([x1, y1, x2, y2], fill=color)

        elif shape_type == "sphere":
            if radius is None:
                raise ValueError("Radius must be provided for a sphere.")
            radius = int(radius * scale_factor)

            # Calculate the maximum allowed coordinates for the shape to be fully inside the canvas
            max_x = image_size[0] - radius - 1
            max_y = image_size[1] - radius - 1
            min_x = radius + 1
            min_y = radius + 1

            # Randomize the center within these bounds
            center = (
                torch.randint(min_x, max_x, (1,)).item(),
                torch.randint(min_y, max_y, (1,)).item(),
            )

            x1, y1 = center[0] - radius, center[1] - radius
            x2, y2 = center[0] + radius, center[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=color)

        else:
            raise ValueError(
                "Invalid shape_type. Use 'cylinder', 'cube', or 'sphere'."
            )

        return image

    def gen_color(self, img, cue_str):
        cue = img.copy()
        cue.paste(self.color_dict[cue_str], [0, 0, cue.size[0], cue.size[1]])
        return cue

    def gen_shape(self, img, cue_str, size=100):
        if self.primitive:
            if cue_str == "cylinder":
                size = (50, 100)
            cue = self.draw_shape(
                image_size=(img.size[0], img.size[1]),
                shape_type=cue_str,
                size=size,
                radius=100,
                color=(255, 255, 255),
            )
        else:
            cue = self.cue_assets[cue_str][self.shape_cue_color].copy()
        return cue

    def gen_conjunction(self, img, cue_str, size=100):
        shape = cue_str[0]
        color = cue_str[1]
        if self.primitive:
            if shape == "cylinder":
                size = (50, 100)
            cue = self.draw_shape(
                image_size=(img.size[0], img.size[1]),
                shape_type=shape,
                size=size,
                radius=100,
                color=color,
            )
        else:
            cue = self.cue_assets[shape][color].copy()

        return cue

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        image_path = self.files[index]
        cue_str = self.cues[index]
        label = self.counts[index]
        mode = self.modes[index]

        img = Image.open(image_path)
        img = img.convert("RGB")

        # create a cue and get the right numerical label
        if mode == "color":
            cue = self.gen_color(img, cue_str)
        elif mode == "shape":
            cue = self.gen_shape(img, cue_str)
        elif mode == "conjunction":
            cue = self.gen_conjunction(img, cue_str)
        else:
            raise ValueError(
                "Invalid mode. Must be 'color', 'shape', or 'conjunction'"
            )

        if self.return_metadata:
            if mode == "conjunction":
                cue_str = f"{cue_str[0]}_{cue_str[1]}"
            return (
                (
                    self.transform(cue),
                    self.transform(img),
                ),
                label,
                image_path,
                mode,
                cue_str,
            )
        else:
            return (self.transform(cue), self.transform(img)), label
