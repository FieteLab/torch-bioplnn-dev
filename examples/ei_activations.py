import os
import sys
import warnings
from traceback import print_exc

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
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


def initialize_dataloader(config, seed):
    if config.dataset == "qclevr":
        _, test_loader = get_qclevr_dataloaders(
            **without_keys(config, ["dataset", "return_metadata"]),
            return_metadata=True,
            seed=seed,
        )
    elif config.dataset == "cabc":
        _, test_loader = get_cabc_dataloaders(
            **without_keys(config, ["dataset"]),
            seed=seed,
        )
    else:
        _, test_loader = get_image_classification_dataloaders(
            **config,
            seed=seed,
        )
    return test_loader


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
    if (
        config.data.dataset == "qclevr" and config.data.val_batch_size != 1
    ) or config.data.batch_size != 1:
        warnings.warn("Batch size is not 1. Setting to 1.")
        if config.data.dataset == "qclevr":
            config.data.val_batch_size = 1
        else:
            config.data.batch_size = 1

    test_loader = initialize_dataloader(config.data, config.seed)

    # Record results
    save_dicts = []
    bar = tqdm(
        test_loader,
        desc="Test",
        disable=not config.tqdm,
    )
    with torch.no_grad():
        for i, sample in enumerate(bar):
            saved_sample = {}
            if i >= config.activations.num_samples:
                break

            # Parse batch
            if config.data.dataset == "qclevr":
                x, label, image_path, mode, cue_str = sample
                saved_sample.update(
                    {
                        "cue": x[0],
                        "mixture": x[1],
                        "image_path": image_path,
                        "mode": mode,
                        "cue_str": cue_str,
                    }
                )
            else:
                x, label = sample
                saved_sample.update({"x": x})
            saved_sample.update({"label": label})

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
                saved_sample.update({"model_outputs": outputs.detach().cpu()})
                if config.data.dataset == "qclevr":
                    saved_sample.update(
                        {
                            "outs_cue": [
                                t.detach().cpu() if t is not None else None
                                for t in outs_cue
                            ],
                            "h_pyrs_cue": [
                                t.detach().cpu() if t is not None else None
                                for t in h_pyrs_cue
                            ],
                            "h_inters_cue": [
                                t.detach().cpu() if t is not None else None
                                for t in h_inters_cue
                            ],
                            "fbs_cue": [
                                t.detach().cpu() if t is not None else None
                                for t in fbs_cue
                            ],
                            "outs_mixture": [
                                t.detach().cpu() if t is not None else None
                                for t in outs_mix
                            ],
                            "h_pyrs_mixture": [
                                t.detach().cpu() if t is not None else None
                                for t in h_pyrs_mix
                            ],
                            "h_inters_mixture": [
                                t.detach().cpu() if t is not None else None
                                for t in h_inters_mix
                            ],
                            "fbs_mixture": [
                                t.detach().cpu() if t is not None else None
                                for t in fbs_mix
                            ],
                        }
                    )
                else:
                    saved_sample.update(
                        {
                            "outs": [
                                t.detach().cpu() if t is not None else None for t in out
                            ],
                            "h_pyrs": [
                                t.detach().cpu() if t is not None else None
                                for t in h_pyr
                            ],
                            "h_inters": [
                                t.detach().cpu() if t is not None else None
                                for t in h_inter
                            ],
                            "fbs": [
                                t.detach().cpu() if t is not None else None for t in fb
                            ],
                        }
                    )

            if config.criterion.all_timesteps:
                pred = outputs[-1].argmax(-1)
            else:
                pred = outputs.argmax(-1)
            saved_sample.update({"pred": pred.detach().cpu()})
            save_dicts.append(saved_sample)

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
    torch.save(save_dicts, save_path)


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
