import os
import sys
from traceback import print_exc

import hydra
import torch
from addict import Dict as AttrDict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

import wandb
from bioplnn.loss import EDLLoss
from bioplnn.models.classifiers import ImageClassifier, QCLEVRClassifier
from bioplnn.utils import (
    get_cabc_dataloaders,
    get_image_classification_dataloaders,
    get_qclevr_dataloaders,
    without_keys,
)


def initialize_model(
    dataset: str, config: AttrDict, exclude_keys: list[str] = []
) -> torch.nn.Module:
    if dataset == "qclevr":
        model = QCLEVRClassifier(**without_keys(config, exclude_keys))
    else:
        model = ImageClassifier(**without_keys(config, exclude_keys))

    return model


def initialize_criterion(fn: str, num_classes=None) -> torch.nn.Module:
    if fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif fn == "edl":
        criterion = EDLLoss(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Criterion {fn} not implemented")

    return criterion


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
    return test_loader


def _test(
    config: AttrDict,
    model: nn.Module,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Perform a single evaluation iteration.

    Args:
        config (AttrDict): The experiment configuration.
        model (nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing the val loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    bar = tqdm(
        test_loader,
        desc="Validation",
        disable=not config.tqdm,
    )
    with torch.no_grad():
        for i, (x, labels) in enumerate(bar):
            if config.debug_forward and i >= 20:
                break
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            labels = labels.to(device)
            # Forward pass
            outputs = model(
                x=x,
                num_steps=config.train.num_steps,
                loss_all_timesteps=config.criterion.all_timesteps,
            )
            if config.criterion.all_timesteps:
                losses = []
                for output in outputs:
                    losses.append(criterion(output, labels))
                loss = sum(losses) / len(losses)
                outputs = outputs[-1]
            else:
                loss = criterion(outputs, labels)

            # Update statistics
            val_loss += loss.item()
            predicted = outputs.argmax(-1)
            correct = (predicted == labels).sum().item()
            val_correct += correct
            val_total += len(labels)

    # Calculate average val loss and accuracy
    val_loss /= len(test_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc


def test(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    config = AttrDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    if config.wandb.group is not None and not config.wandb.group.endswith("test"):
        config.wandb.group = f"{config.wandb.group}_test"
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

    # Initialize the loss function
    criterion = initialize_criterion(
        config.criterion.fn, num_classes=config.model.num_classes
    )

    # Initialize Test Dataloader
    test_loader = initialize_dataloader(
        config.data, config.model.rnn_kwargs.in_size, config.seed
    )

    # Compute loss and accuracy on test data
    loss, acc = _test(config, model, criterion, test_loader, device)

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
    config_path="/om2/user/valmiki/bioplnn/config/ei_crnn",
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
