import os
import sys
import warnings
from traceback import print_exc

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from bioplnn.loss import EDLLoss
from bioplnn.models.classifiers import ImageClassifier, QCLEVRClassifier
from bioplnn.utils import (
    AttrDict,
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
        _, test_loader = get_qclevr_dataloaders(
            **without_keys(config, ["dataset"]),
            resolution=resolution,
            return_image_metadata=True,
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
                return_activations=False,
            )
            if config.criterion.all_timesteps:
                logits = outputs.permute(1, 2, 0)
                labels = labels.unsqueeze(-1).expand(-1, outputs.shape[0])
            else:
                logits = outputs

            # Compute the loss
            loss = criterion(logits, labels)

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
    if config.data.batch_size != 1:
        warnings.warn("Batch size is not 1. Setting to 1.")
        config.data.batch_size = 1
    test_loader = initialize_dataloader(
        config.data, config.model.rnn_kwargs.in_size, config.seed
    )

    # Record results
    if config.data.dataset == "qclevr":
        cues = []
        mixtures = []
        image_paths = []
        modes = []
        cue_strs = []
        if config.activations.save_activations:
            outs_cues = []
            h_inters_cues = []
            h_pyrs_cues = []
            fbs_cues = []
            outs_mixtures = []
            h_inters_mixtures = []
            h_pyrs_mixtures = []
            fbs_mixtures = []
    else:
        xs = []
        if config.activations.save_activations:
            outs = []
            h_pyrs = []
            h_inters = []
            fbs = []
    model_outputs = []
    labels = []
    preds = []
    bar = tqdm(
        test_loader,
        desc="Test",
        disable=not config.tqdm,
    )
    with torch.no_grad():
        for i, sample in enumerate(bar):
            if i >= config.activations.num_samples:
                break

            # Parse batch
            if config.data.dataset == "qclevr":
                x, label, image_path, mode, cue_str = sample
                cues.append(x[0])
                mixtures.append(x[1])
                image_paths.append(image_path)
                modes.append(mode)
                cue_strs.append(cue_str)
            else:
                x, label = sample
                xs.append(x)
            labels.append(label)

            # Move to device
            try:
                x = x.to(device)
            except AttributeError:
                x = [t.to(device) for t in x]
            label = label.to(device)
            # Forward pass
            out = model(
                x=x,
                num_steps=config.train.num_steps,
                loss_all_timesteps=config.criterion.all_timesteps,
                return_activations=True,
            )
            if config.data.dataset == "qclevr":
                (
                    outputs,
                    outs_cue,
                    h_pyrs_cue,
                    h_inters_cue,
                    fbs_cue,
                    outs_mix,
                    h_pyrs_mix,
                    h_inters_mix,
                    fbs_mix,
                ) = out
            else:
                outputs, out, h_pyr, h_inter, fb = out

            if config.activations.save_activations:
                model_outputs.append(outputs.detach().cpu())
                if config.data.dataset == "qclevr":
                    outs_cues.append(
                        [t.detach().cpu() if t is not None else None for t in outs_cue]
                    )
                    h_pyrs_cues.append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_pyrs_cue
                        ]
                    )
                    h_inters_cues.append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_inters_cue
                        ]
                    )
                    fbs_cues.append(
                        [t.detach().cpu() if t is not None else None for t in fbs_cue]
                    )
                    outs_mixtures.append(
                        [t.detach().cpu() if t is not None else None for t in outs_mix]
                    )
                    h_pyrs_mixtures.append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_pyrs_mix
                        ]
                    )
                    h_inters_mixtures.append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_inters_mix
                        ]
                    )
                    fbs_mixtures.append(
                        [t.detach().cpu() if t is not None else None for t in fbs_mix]
                    )
                else:
                    outs.append(
                        [t.detach().cpu() if t is not None else None for t in out]
                    )
                    h_pyrs.append(
                        [t.detach().cpu() if t is not None else None for t in h_pyr]
                    )
                    h_inters.append(
                        [t.detach().cpu() if t is not None else None for t in h_inter]
                    )
                    fbs.append(
                        [t.detach().cpu() if t is not None else None for t in fb]
                    )

            if config.criterion.all_timesteps:
                pred = outputs[-1].argmax(-1)
            else:
                pred = outputs.argmax(-1)
            preds.append(pred.detach().cpu())

    save_dict = {}
    if config.data.dataset == "qclevr":
        save_dict.update(
            {
                "cues": cues,
                "mixtures": mixtures,
                "image_paths": image_paths,
                "modes": modes,
                "cue_strs": cue_strs,
            }
        )
        if config.activations.save_activations:
            save_dict.update(
                {
                    "model_outputs": model_outputs,
                    "outs_cue": outs_cues,
                    "h_pyrs_cue": h_pyrs_cues,
                    "h_inters_cue": h_inters_cues,
                    "fbs_cue": fbs_cues,
                    "outs_mixture": outs_mixtures,
                    "h_pyrs_mixture": h_pyrs_mixtures,
                    "h_inters_mixture": h_inters_mixtures,
                    "fbs_mixture": fbs_mixtures,
                }
            )
    else:
        save_dict.update({"xs": xs})
        if config.activations.save_activations:
            save_dict.update(
                {
                    "model_outputs": model_outputs,
                    "outs": outs,
                    "h_pyrs": h_pyrs,
                    "h_inters": h_inters,
                    "fbs": fbs,
                }
            )
    save_dict.update({"labels": labels, "preds": preds})

    # Save activations
    if config.data.dataset == "qclevr":
        save_dir = os.path.join(
            config.activations.save_root,
            config.data.dataset,
            config.checkpoint.run,
            config.data.mode,
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            "activations.pt" if config.activations.save_activations else "summary.pt",
        )
    else:
        save_dir = os.path.join(
            config.activations.save_root,
            config.data.dataset,
            config.checkpoint.run,
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            "activations.pt" if config.activations.save_activations else "summary.pt",
        )
    torch.save(save_dict, save_path)


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
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()


# python -u src/bioplnn/trainers/ei_tester.py data.mode=every data.holdout=[] model.modulation_type=ag checkpoint.run=skilled-spaceship-116
