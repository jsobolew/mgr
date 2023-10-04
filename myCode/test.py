import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="configs/experiments", config_name="AlexNetTaskILNoise")
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))