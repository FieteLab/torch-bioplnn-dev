import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class Mazes(Dataset):
    def __init__(
        self, root, train=True, subset=1.0, return_metadata=False, transform=None
    ):
        if train:
            self.annotations_path = os.path.join(root, "mazes_train.json")
            self.mazes_path = os.path.join(root, "mazes_train.npy")
        else:
            self.annotations_path = os.path.join(root, "mazes_val.json")
            self.mazes_path = os.path.join(root, "mazes_val.npy")
        self.subset = subset
        self.return_metadata = return_metadata
        self.frame_thickness = 3
        self.annotation_scaling = 3  # original mazes were labeled on a 144x144 grid, but we want to work on the original 48x48 grid
        self.cue_size = (
            6 // self.annotation_scaling
        )  # cues are 6x6 pixels in the 144x144 grid (2x2 in the 48x48 grid)

        self.transform = transform

        # Read list of stimulus records
        self.data = self.read_annotations(self.annotations_path, self.subset)

        # Load matrix of mazes
        self.mazes = np.load(self.mazes_path)

    def read_annotations(self, file, subset):
        with open(file, "rb") as f:
            data = json.load(f)

        annotations = data["serrelab_anns"]
        if subset != 1:
            pos = [x for x in annotations if x["same"] == 1]
            neg = [x for x in annotations if x["same"] == 0]
            annotations = random.sample(pos, int(subset * len(pos))) + random.sample(
                neg, int(subset * len(neg))
            )
        data["serrelab_anns"] = annotations

        return data

    def __len__(self):
        return len(self.data["serrelab_anns"])

    def tensor_to_image(self, img_w_dots, draw_cues=True):
        img = img_w_dots.detach().cpu().numpy()
        dots = img[3, :, :] * 255
        img = np.transpose(img[0:3, :, :], (1, 2, 0)) * 255
        img = img.astype("int32")
        if draw_cues:
            img[dots == 255] = (
                255,
                0,
                0,
            )
        return img

    def __getitem__(self, index: int):
        record = self.data["serrelab_anns"][index]
        record["dataset_index"] = index
        # Get base maze
        maze = self.mazes[record["id"]].copy()
        maze = np.transpose(maze, (1, 2, 0))
        # Remove red and green points from original data
        y_values_red, x_values_red = np.where(np.all(maze == (1, 0, 0), axis=-1))
        y_values_green, x_values_green = np.where(np.all(maze == (0, 1, 0), axis=-1))

        maze[y_values_red, x_values_red, :] = (1.0, 1.0, 1.0)
        maze[y_values_green, x_values_green, :] = (1.0, 1.0, 1.0)

        # Crop and pad
        maze = maze[
            record["row_start"] : record["row_stop"],
            record["col_start"] : record["col_stop"],
            :,
        ]
        maze = np.pad(
            maze,
            (
                (self.frame_thickness, self.frame_thickness),
                (self.frame_thickness, self.frame_thickness),
                (0, 0),
            ),
        )

        # Flips and rotations
        if record["horizontal_flip"]:
            maze = np.fliplr(maze)
        if record["vertical_flip"]:
            maze = np.flipud(maze)

        maze = np.rot90(maze, k=record["rotation_k"])

        # Insert blocks (extra walls)
        maze[record["blocks_y"], record["blocks_x"], :] = (
            0,
            0,
            0,
        )  # note that they will be in original resolution

        # Resize
        # maze = cv2.resize(
        #     maze * 255,
        #     dsize=(record["target_size"], record["target_size"]),
        #     interpolation=cv2.INTER_NEAREST,
        # )
        # maze = maze / 255

        # Add cue channel
        dot_channel = np.zeros_like(maze[:, :, 0])[:, :, np.newaxis]
        dot_channel[
            record["fixation_top_y"] // self.annotation_scaling : record[
                "fixation_top_y"
            ]
            // self.annotation_scaling
            + self.cue_size,
            record["fixation_left_x"] // self.annotation_scaling : record[
                "fixation_left_x"
            ]
            // self.annotation_scaling
            + self.cue_size,
            :,
        ] = 1
        dot_channel[
            record["cue_top_y"] // self.annotation_scaling : record["cue_top_y"]
            // self.annotation_scaling
            + self.cue_size,
            record["cue_left_x"] // self.annotation_scaling : record["cue_left_x"]
            // self.annotation_scaling
            + self.cue_size,
            :,
        ] = 1
        maze = np.concatenate([maze, dot_channel], axis=2)

        maze = torch.Tensor(maze).permute(2, 0, 1)

        # Transform
        if self.transform is not None:
            maze = self.transform(maze)

        # Gather
        if self.return_metadata:
            return {
                "image": maze,
                "label": record["same"],
                "id": record["id"],
                "index": record["dataset_index"],
                "cue_x": record["cue_x"],
                "cue_y": record["cue_y"],
                "fixation_x": record["fixation_x"],
                "fixation_y": record["fixation_y"],
            }

        return maze, record["same"]

        # except Exception as e:
        #     print(
        #         f"Error in getting sample with index {record['dataset_index']}, id {record['id']}, and serre_lab sample {record['serrelab_sample']}: {str(e)}"
        #     )
        #     print("Sampling new random image")
        #     new_idx = random.choice(list(range(len(self))))
        #     return self.__getitem__(new_idx)
