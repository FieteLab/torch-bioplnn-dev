import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None,
    config_path="/om2/user/valmiki/bioplnn/config/crnn",
    config_name="config",
)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    print(OmegaConf.to_yaml(HydraConfig.get()))


if __name__ == "__main__":
    main()
