import os
import sys
import warnings
from collections import defaultdict
from traceback import print_exc

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from bioplnn.utils import AttrDict, initialize_dataloader, initialize_model


def test(dict_config: DictConfig) -> None:
    config = OmegaConf.to_container(dict_config, resolve=True)
    config = AttrDict(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load checkpoint
    model = initialize_model(**config.model).to(device)
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

    if (
        config.data.dataset != "correlated_dots"
        and config.activations.shuffle_test
    ):
        config.data.shuffle_test = True

    # Get the data loaders
    _, test_loader = initialize_dataloader(seed=config.seed, **config.data)

    # Record results
    save_dict = defaultdict(list)
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
                save_dict["cue"].append(x[0])
                save_dict["mixture"].append(x[1])
                save_dict["image_path"].append(image_path)
                save_dict["mode"].append(mode)
                save_dict["cue_str"].append(cue_str)
            else:
                x, label = sample
                save_dict["x"].append(x)
            save_dict["label"].append(label)

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
                outputs, outs, h_pyrs, h_inters, fbs = out

            if config.activations.save_activations:
                save_dict["model_outputs"].append(outputs.detach().cpu())
                if config.data.dataset == "qclevr":
                    # Cue
                    save_dict["outs_cue"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in outs_cue
                        ]
                    )
                    save_dict["h_pyrs_cue"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_pyrs_cue
                        ]
                    )
                    if h_inters_cue is not None:
                        save_dict["h_inters_cue"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in h_inters_cue
                            ]
                        )
                    if fbs_cue is not None:
                        save_dict["fbs_cue"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in fbs_cue
                            ]
                        )

                    # Mix
                    save_dict["outs_mix"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in outs_mix
                        ]
                    )
                    save_dict["h_pyrs_mix"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_pyrs_mix
                        ]
                    )
                    if h_inters_mix is not None:
                        save_dict["h_inters_mix"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in h_inters_mix
                            ]
                        )
                    if fbs_mix is not None:
                        save_dict["fbs_mix"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in fbs_mix
                            ]
                        )
                else:
                    save_dict["outs"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in outs
                        ]
                    )
                    save_dict["h_pyrs"].append(
                        [
                            t.detach().cpu() if t is not None else None
                            for t in h_pyrs
                        ]
                    )
                    if h_inters is not None:
                        save_dict["h_inters"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in h_inters
                            ]
                        )
                    if fbs is not None:
                        save_dict["fbs"].append(
                            [
                                t.detach().cpu() if t is not None else None
                                for t in fbs
                            ]
                        )

            if config.criterion.all_timesteps:
                pred = outputs[-1].argmax(-1)
            else:
                pred = outputs.argmax(-1)
            save_dict["pred"].append(pred.detach().cpu())

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
            "activations.pt"
            if config.activations.save_activations
            else "summary.pt",
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
            "activations.pt"
            if config.activations.save_activations
            else "summary.pt",
        )
    save_dict = dict(save_dict)
    torch.save(save_dict, save_path)


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
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()


# python -u src/bioplnn/trainers/ei_tester.py data.mode=every data.holdout=[] model.modulation_type=ag checkpoint.run=skilled-spaceship-116
