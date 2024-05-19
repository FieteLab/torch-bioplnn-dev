import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bioplnn.datasets.ei import qCLEVRDataset
from bioplnn.models.ei import Conv2dEIRNN
from bioplnn.utils import AttrDict, rescale


@hydra.main(
    version_base=None,
    config_path="/om2/user/valmiki/bioplnn/config",
    config_name="config_ei_test",
)
def test(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colors = [
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
    shapes = ["cube", "cylinder", "sphere"]
    conjunctions = [
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

    checkpoint_path = os.path.join(
        config.checkpoint.root,
        config.checkpoint.run,
        (
            "checkpoint.pt"
            if config.checkpoint.epoch is None
            else f"checkpoint_{config.checkpoint.epoch}.pt"
        ),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Conv2dEIRNN(**config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval()

    if config.data.mode == "colors":
        cues = colors
    elif config.data.mode == "shapes":
        cues = shapes
    elif config.data.mode == "conjunction":
        cues = conjunctions
    elif config.data.mode == "every":
        cues = colors + shapes + conjunctions
    else:
        raise ValueError(f"Invalid mode {config.data.mode}")

    if len(config.data.holdout) > 0:
        in_dist_holdouts = [c for c in cues if c not in config.data.holdout]
        out_dist_holdouts = config.data.holdout
    else:
        in_dist_holdouts = []
        out_dist_holdouts = None
    for dist in ("in_dist", "out_dist"):
        if dist == "in_dist":
            holdout = in_dist_holdouts
        else:
            if out_dist_holdouts is None:
                continue
            holdout = out_dist_holdouts

        clevr_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(rescale),
                transforms.Resize(config.model.input_size),
            ]
        )
        val_dataset = qCLEVRDataset(
            data_root=config.data.root,
            assets_path=config.data.assets_path,
            clevr_transforms=clevr_transforms,
            return_images=config.save_results,
            split="valid",
            holdout=holdout,
            mode=config.data.mode,
            primitive=config.data.primitive,
            num_workers=config.data.num_workers,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        accs = []
        labels = []
        preds = []
        image_paths = []
        modes = []
        for batch in tqdm(val_dataloader):
            cue = batch[0].to(device)
            mixture = batch[1].to(device)
            label = batch[2]

            # get the model preds
            out = model(cue, mixture)
            pred = torch.argmax(out, axis=-1).cpu()
            preds.extend(pred.numpy().tolist())
            labels.extend(label.numpy().tolist())
            accs.extend((pred == label).long().numpy().tolist())
            if config.save_results:
                image_paths.extend(batch[3])
                modes.extend(batch[4])

        if config.save_results:
            os.makedirs(config.save_results_root, exist_ok=True)
            save_path = os.path.join(
                config.save_results_root,
                f"{config.checkpoint.run}_{dist}.json",
            )
            with open(save_path, "w") as f:
                json.dump(
                    {
                        "preds": preds,
                        "labels": labels,
                        "accs": accs,
                        "image_paths": image_paths,
                        "modes": modes,
                    },
                    f,
                )

        print(f"Dist: {dist}, Acc: {np.mean(accs)}, Std: {np.std(accs)}")


if __name__ == "__main__":
    test()

# python -u src/bioplnn/trainers/ei_tester.py data.mode=every data.holdout=[] model.modulation_type=ag checkpoint.run=skilled-spaceship-116
