import argparse
from typing import Optional

import numpy as np
import torch
import tqdm
from method import attnCNNMethod
from params import BaselineAttentionParams
from bioplnn.models.ei import 
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import rescale

from bioplnn.data import qCLEVRDataModule, qCLEVRDataset


def main(args, params: Optional[BaselineAttentionParams] = None):
    if params is None:
        params = BaselineAttentionParams()


    for mode in ["color", "shape", "conjunction"]:
        holdout_dict = {
            "color": [
                ["out_dist", ["blue", "green"]],
                [
                    "in_dist",
                    [
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
                    ],
                ],
            ],
            "shape": [["out_dist", ["cube"]], ["in_dist", ["cylinder", "sphere"]]],
            "conjunction": [
                ["out_dist", ["cube_blue", "cube_green"]],
                [
                    "in_dist",
                    [
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
                    ],
                ],
            ],
        }
        for holdout_name, holdout in holdout_dict[mode]:
            if mode == "color":
                ckpt = "vis_attention/l4kfni5a/checkpoints/epoch=98-step=36927.ckpt"
            elif mode == "shape":
                ckpt = "vis_attention/38mv67dx/checkpoints/epoch=98-step=32967.ckpt"
            elif mode == "conjunction":
                ckpt = "vis_attention/vngfuv3x/checkpoints/epoch=98-step=45144.ckpt"
        
            
            clevr_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(rescale),
                    transforms.Resize(params.resolution),
                ]
            )
            val_dataset = qCLEVRDataset(
                data_root=params.data_root,
                clevr_transforms=clevr_transforms,
                split="valid",
                holdout=holdout,
                mode=mode,
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=params.val_batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True,
            )
            # clevr_datamodule = qCLEVRDataModule(
            #     data_root=params.data_root,
            #     train_batch_size=params.batch_size,
            #     val_batch_size=params.val_batch_size,
            #     clevr_transforms=clevr_transforms,
            #     num_workers=params.num_workers,
            #     holdout=holdout,
            #     mode=mode,
            # )
            try:
                model = attnCNNMethod.load_from_checkpoint(
                    ckpt, model=model, params=params, datamodule=None, cue=args.cue
                )
            except Exception as e:
                print(f"Error loading Model: {model_class} | Exception: {e}")
                continue
            model = model.to("cuda")
            model = model.eval()

            accs = []
            for batch in val_dataloader:
                labels = batch[2]

                # get the model preds
                out = model([batch[0].to("cuda"), batch[1].to("cuda")])
                preds = torch.argmax(out, axis=-1).cpu()
                accs = accs + (preds == labels).long().numpy().tolist()

            print(
                f"Model: {model_class}, Mode: {mode}, Holdout: {holdout_name}, Acc: {np.mean(accs)}, Std: {np.std(accs)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logname", help="prefix for the logger", type=str)
    parser.add_argument(
        "--holdout", help="keep attribute out of training", nargs="+", default=[]
    )
    parser.add_argument(
        "--mode",
        help="cue specification",
        type=str,
        default="color",
        choices=["color", "shape", "conjunction"],
    )
    parser.add_argument(
        "--cue",
        help="task mode",
        type=str,
        default=None,
        choices=["explicit", "implicit"],
    )
    parser.add_argument(
        "--model",
        help="model type",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "attnCNNImplicit",
            "resnetImplicit",
            "txferImplicitVIT",
            "txferImplicitSwinT",
        ],
    )
    parser.add_argument(
        "--visualize", help="visualize the dataset", action="store_true", default=False
    )
    args = parser.parse_args()

    main(args)
