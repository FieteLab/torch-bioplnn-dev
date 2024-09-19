import os
import sys
from traceback import print_exc

import hydra
import torch
import wandb
from addict import Dict as AttrDict
from ei_trainer import val_epoch
from omegaconf import DictConfig, OmegaConf

from bioplnn.models.classifiers import ImageClassifier, QCLEVRClassifier
from bioplnn.utils import (
    get_cabc_dataloaders,
    get_image_classification_dataloaders,
    get_qclevr_dataloaders,
    without_keys,
)


def initialize_model(
    dataset: str, config: AttrDict, exclude_keys: list[str]
) -> torch.nn.Module:
    if dataset == "qclevr":
        model = QCLEVRClassifier(**without_keys(config, exclude_keys))
    else:
        model = ImageClassifier(**without_keys(config, exclude_keys))

    return model


def initialize_dataloader(config, resolution, seed):
    if config.dataset == "qclevr":
        _, _, test_loader = get_qclevr_dataloaders(
            **without_keys(config, ["dataset", ""]),
            resolution=resolution,
            seed=seed,
        )
    elif config.dataset == "cabc":
        _, test_loader = get_cabc_dataloaders(
            **without_keys(config, ["dataset"]),
            resolution=resolution,
            seed=seed,
        )
    else:
        _, test_loader = get_image_classification_dataloaders(
            **config,
            resolution=resolution,
            seed=seed,
        )


def test(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    if config.wandb.group is not None and not config.wandb.group.endswith("test"):
        config.wandb.group = f"{config.wandb}_test"
    wandb.require("core")
    wandb.init(
        config=config,
        settings=wandb.Settings(start_method="thread"),
        **config.wandb,
    )

    # Initialize model and load checkpoint
    model = initialize_model(dataset=config.data.dataset, config=config.model).to(
        device
    )
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval()

    # Initialize Test Dataloader
    test_loader = initialize_dataloader(
        config.data, config.model.rnn_kwargs.in_size, config.seed
    )

    # Compute loss and accuracy on test data
    loss, acc = val_epoch(config, model, test_loader, device)

    # Record results
    print(f"Loss: {loss}, Acc: {acc}")
    wandb.log(dict(loss=loss, acc=acc))

    # for batch in tqdm(test_loader):
    #     cue = batch[0].to(device)
    #     mixture = batch[1].to(device)
    #     label = batch[2]

    #     # get the model preds
    #     out, (outs_cue, outs_mixture) = model(cue, mixture, return_layer_outputs=True)
    #     for t in range(len(outs_cue)):
    #         for i in range(len(outs_cue[t])):
    #             outs_cue[t][i] = outs_cue[t][i].detach().cpu()
    #             outs_mixture[t][i] = outs_mixture[t][i].detach().cpu()

    #     cues.append(batch[0])
    #     mixtures.append(batch[1])
    #     outs_cues.append(outs_cue)
    #     outs_mixtures.append(outs_mixture)
    #     i += 1
    #     if i == 50:
    #         break

    # save_path = os.path.join(
    #     config.save_activations_root,
    #     f"{config.checkpoint.run}_{config.data.mode}.json",
    # )
    # torch.save(
    #     {
    #         "cues": cues,
    #         "mixtures": mixture,
    #         "outs_cue": outs_cues,
    #         "outs_mixture": outs_mixtures,
    #     },
    #     save_path,
    # )
    # if config.get_accuracy or config.save_results:
    #     accs = []
    #     labels = []
    #     preds = []
    #     image_paths = []
    #     modes = []
    #     cues = []
    #     for batch in tqdm(val_dataloader):
    #         cue = batch[0].to(device)
    #         mixture = batch[1].to(device)
    #         label = batch[2]

    #         # get the model preds
    #         out = model(cue, mixture)
    #         pred = torch.argmax(out, axis=-1).cpu()
    #         preds.extend(pred.numpy().tolist())
    #         labels.extend(label.numpy().tolist())
    #         accs.extend((pred == label).long().numpy().tolist())
    #         if config.save_results:
    #             image_paths.extend(batch[3])
    #             modes.extend(batch[4])
    #             cues.extend(batch[5])

    #     if config.save_results:
    #         os.makedirs(config.save_results_root, exist_ok=True)
    #         save_path = os.path.join(
    #             config.save_results_root,
    #             f"{config.checkpoint.run}_{dist}.json",
    #         )
    #         with open(save_path, "w") as f:
    #             json.dump(
    #                 {
    #                     "preds": preds,
    #                     "labels": labels,
    #                     "accs": accs,
    #                     "image_paths": image_paths,
    #                     "modes": modes,
    #                     "cues": cues,
    #                 },
    #                 f,
    #             )


@hydra.main(
    version_base=None,
    config_path="/om2/user/valmiki/bioplnn/config",
    config_name="config",
)
def main(config: DictConfig):
    try:
        test(config)
    except Exception as e:
        if config.debug_level > 0:
            print_exc(file=sys.stderr)
        else:
            print(e, file=sys.stderr)
        wandb.log(dict(error=str(e)))
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()


# python -u src/bioplnn/trainers/ei_tester.py data.mode=every data.holdout=[] model.modulation_type=ag checkpoint.run=skilled-spaceship-116
