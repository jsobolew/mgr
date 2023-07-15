import hydra
# from omegaconf import DictConfig, OmegaConf
import torch
from Nets import SmallAlexNetTaslIL
from taskIL import train_validation_all_classes
from dataloaders.cifar10 import cifar10_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining
import numpy as np
import torch.optim as optim
import wandb
import torch.nn.functional as F
from utils import get_device

model_dict = {
    "SmallAlexNetTaslIL" : SmallAlexNetTaslIL
}



@hydra.main(version_base=None, config_path="configs/experiments", config_name="AlexNetTaskILNoise")
def main(cfg ) -> None:
    device = get_device()

    # ------
    classes = np.arange(10)
    np.random.shuffle(classes)
    classes = list(classes)

    task1 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task2 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task3 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task4 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task5 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())

    tasks = [task1 ,task2, task3, task4, task5]
    # ------

    rehersal_loader = dataloader_pretraining(cfg['rehersal_dataset'], no_classes=2)

    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg['project'],
        
        # track hyperparameters and run metadata
        config={
        "setup": cfg['setup'],
        "learning_rate": cfg['learning_rate'],
        "architecture": cfg['architecture'],
        "dataset": cfg['dataset'],
        "epochs": cfg['epochs'],
        },
        mode="disabled"
    )

    model_reference = model_dict[cfg['architecture']]
    model = model_reference(feat_dim=10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    train_validation_all_classes(model, optimizer, tasks, device, rehersal_loader, epoch=cfg['epochs'], log_interval = 10)

if __name__ == "__main__":
    main()