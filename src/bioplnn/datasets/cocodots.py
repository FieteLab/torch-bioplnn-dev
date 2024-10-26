import copy
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDots(Dataset):

    def __init__(self, anns_file, img_dir, size=(150, 150), subset=1, shuffle=False, stimulus_condition='WhiteOutline'):
        self.anns_file = anns_file
        self.img_dir = img_dir
        self.size = size  # (h, w)
        self.stimulus_condition = stimulus_condition
        self.transform = ToTorchFormatTensor(div=True, aug=False)
        self.shuffle = shuffle
        self.subset = subset
        self.data = self.read_anns(anns_file, shuffle=self.shuffle, subset=self.subset)
        self.create_index()

    def read_anns(self, file, shuffle, subset):
        with open(file, 'rb') as f:
            data = json.load(f)

        serrelab_anns = data["serrelab_anns"]
        if subset != 1:
            pos = [x for x in serrelab_anns if x["same"]==1]
            neg = [x for x in serrelab_anns if x["same"]==0]
            serrelab_anns = random.sample(pos, int(subset*len(pos))) + random.sample(neg, int(subset*len(neg)))

        if shuffle:
            random.shuffle(serrelab_anns)

        data["serrelab_anns"] = serrelab_anns

        return data

    def create_index(self):
        '''
        Adjuted from:
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py,
        '''
        # create index
        print('creating index...')
        cats, imgs, img_to_anns = {}, {}, {}
        img_to_serrelab_anns = defaultdict(list)

        if 'annotations' in self.data:
            for ann in self.data['annotations']:
                img_to_anns[ann['image_id']] = ann

        if 'images' in self.data:
            for img in self.data['images']:
                imgs[img['id']] = img

        if 'categories' in self.data:
            for cat in self.data['categories']:
                cats[cat['id']] = cat

        if 'serrelab_anns' in self.data:
            for serrelab_ann in self.data["serrelab_anns"]:
                img_to_serrelab_anns[serrelab_ann["image_id"]].append(serrelab_ann)

        print('index created!')

        # create class members
        self.img_to_anns = img_to_anns
        self.img_to_serrelab_anns = img_to_serrelab_anns
        self.imgs = imgs
        self.cats = cats

    def __len__(self):
        return len(self.data["serrelab_anns"])

    def __getitem__(self, idx):
        try:
            # Get image and annotations
            serrelab_ann = self.data["serrelab_anns"][idx]
            img_data = self.imgs[serrelab_ann["image_id"]]
            ann = self.img_to_anns[serrelab_ann["image_id"]]
            img = Image.open(os.path.join(self.img_dir, img_data["file_name"])).convert('RGB')
            img = np.array(img)

            if 'dot_res_height' in serrelab_ann:
                # Resolution with respect to which the dot coordinates were defined
                dots_height = serrelab_ann['dot_res_height']
                dots_width = serrelab_ann['dot_res_width']
            else:
                dots_height = img_data['height']
                dots_width = img_data['width']

            # Compute rescale factors
            dots_h_factor = self.size[0] / dots_height  # factors to rescale dot coordinates
            dots_w_factor = self.size[1] / dots_width

            h_factor = self.size[0] / img_data['height']  # factors to rescale image
            w_factor = self.size[1] / img_data['width']

            # Find resized dot locations
            serrelab_ann_resize = copy.deepcopy(serrelab_ann)
            serrelab_ann_resize["cue_xy"] = [round(serrelab_ann_resize["cue_xy"][0] * dots_w_factor),
                                             round(serrelab_ann_resize["cue_xy"][1] * dots_h_factor)]
            serrelab_ann_resize["fixation_xy"] = [round(serrelab_ann_resize["fixation_xy"][0] * dots_w_factor),
                                                  round(serrelab_ann_resize["fixation_xy"][1] * dots_h_factor)]

            # Convert to an outline stimulus (only draw outlines for 'things')
            things = [segm for segm in ann["segments_info"] if self.cats[segm["category_id"]]["isthing"] == 1]
            thickness = int(1 / min(h_factor, w_factor))
            segm_contours = [segm["contours_LG"] for segm in things]
            img = make_stimulus(img, segm_contours, thickness=thickness, condition=self.stimulus_condition)

            # Resize image to desired size
            img = Image.fromarray(img).resize((self.size[1], self.size[0]))  # PIL wants w,h
            img = np.array(img)

            # Add dot channel (all 0, except for dot coordinates, which are 255)
            dot_channel = np.zeros_like(img[:, :, 0])
            dot_channel[serrelab_ann_resize["fixation_xy"][1], serrelab_ann_resize["fixation_xy"][0]] = 255
            dot_channel[serrelab_ann_resize["cue_xy"][1], serrelab_ann_resize["cue_xy"][0]] = 255
            img = np.concatenate([img, np.expand_dims(dot_channel, 2)], axis=2)

            # Get label
            label = serrelab_ann["same"]

            # Transform
            img = self.transform(img)

            # Gather
            return_dict = {
                "image": img,
                "label": label,
                "id": img_data["id"],
                "index": idx,
                "cue_x": serrelab_ann_resize["cue_xy"][0],
                "cue_y": serrelab_ann_resize["cue_xy"][1],
                "fixation_x": serrelab_ann_resize["fixation_xy"][0],
                "fixation_y": serrelab_ann_resize["fixation_xy"][1]
            }

            return return_dict

        except Exception as e:
            print(
                "Error in getting sample with index {}, image_id {}, and serre_lab sample {}: {}".format(idx, img_data[
                    "id"], serrelab_ann["serrelab_sample"], str(e)))
            print("Sampling new random image")
            new_idx = random.choice(list(range(len(self))))
            return_dict = self.__getitem__(new_idx)
            return return_dict

    def tensor_to_image(self, img_w_dots, draw_dots=True):
        img = img_w_dots.detach().cpu().numpy()
        dots = img[3, :, :] * 255
        img = np.transpose(img[0:3, :, :], (1, 2, 0)) * 255
        img = img.astype('int32')
        img = img.copy()
        if draw_dots:
            y_values, x_values = np.where(dots != 0)
            dot_positions = list(zip(x_values.tolist(), y_values.tolist()))
            for dot_position in dot_positions:
                img = cv2.circle(img, dot_position, radius=3, color=(255, 255, 255), thickness=cv2.FILLED)
        return img


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    From: https://github.com/c-rbp/pathfinder_experiments/blob/main/utils/transforms.py
    """

    def __init__(self, div=True, aug=True):
        self.div = div
        self.aug = aug

    def __call__(self, pic):
        if self.aug:
            # handle numpy array
            # img = torch.movedim(torch.from_numpy(pic), -1, 0).contiguous()
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # img = torch.movedim(torch.from_numpy(pic), -1, 0).contiguous()
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()

        return img.float().div(255) if self.div else img.float()


def make_stimulus(image, segm_contours, thickness=2, condition='WhiteOutline'):
    assert condition in ["Color_NoBackground", "WhiteOutline", "BlackOutline", "Color+Background"]

    # Make a template image to indicate where the objects and their outlines are
    template_all = np.zeros_like(image) + 255  # initialize the template
    template_all = template_all.astype('int32')

    for contours in segm_contours:
        # Painter's Algorithm (fill first)
        template_all = cv2.drawContours(template_all, [np.array(contour).astype('int32') for contour in contours],
                                        contourIdx=-1, color=(100, 100, 100),
                                        thickness=cv2.FILLED)
        template_all = cv2.drawContours(template_all, [np.array(contour).astype('int32') for contour in contours],
                                        contourIdx=-1, color=(0, 0, 0),
                                        thickness=thickness)

    # Use the template to turn the image into a stimulus of the right condition
    stimulus = image.copy()
    if condition == "Color_NoBackground":
        stimulus[template_all == 255] = 255
    elif condition == "WhiteOutline":
        stimulus[template_all == 0] = 255
        stimulus[template_all != 0] = 0
    elif condition == "BlackOutline":
        stimulus[template_all == 0] = 0
        stimulus[template_all != 0] = 255

    return stimulus
