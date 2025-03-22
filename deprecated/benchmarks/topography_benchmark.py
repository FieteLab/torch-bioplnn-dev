import sys
import time
from pathlib import Path
from traceback import print_exc

import hydra
import pandas as pd
import torch
from addict import Dict as AttrDict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from bioplnn.models import TopographicalRNN
from bioplnn.optimizers import SparseSGD
from bioplnn.utils import dict_flatten, get_benchmark_dataloaders, seed


def test(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hydra_config = HydraConfig.get()
    print(f"Job Num: {hydra_config.job.num}")
    config_dict = OmegaConf.to_container(
        config,
        resolve=True,
    )
    config = AttrDict(config_dict)
    config.freeze()
    save_dir = Path(config.benchmark.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "benchmark1.csv"

    seed(config.seed)
    model = TopographicalRNN(**config.model).to(device)

    if config.model.mm_function == "torch_sparse":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    else:
        optimizer = SparseSGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
        )
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, _ = get_benchmark_dataloaders(**config.data)

    for i, (images, labels) in enumerate(train_loader):
        if i == config.benchmark.warmup_iters:
            start_time = time.time()
        elif i == config.benchmark.warmup_iters + config.benchmark.num_iters:
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            break
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_time = total_time / config.benchmark.num_iters

    series_config = pd.Series(
        {"avg_time": avg_time} | dict_flatten(config, delimiter=".")
    )
    df_config = pd.DataFrame([series_config])
    df_config.to_csv(
        save_path,
        mode="a",
        header=not save_path.exists(),
        index=False,
    )
    print(f"Avg time: {avg_time:.4f} s")
    print("-" * 80)


@hydra.main(
    config_path="/home/valmiki/om2/bioplnn/config/topography",
    config_name="config",
    version_base=None,
)
def main(config: DictConfig):
    try:
        test(config)
    except Exception as e:
        if config.debug_level > 1:
            print_exc(file=sys.stderr)
        else:
            print(e, file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
