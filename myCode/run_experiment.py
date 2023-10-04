import hydra
import torchvision
from omegaconf import OmegaConf
from Nets import SmallAlexNetTaslIL, ResNet18IL
from myCode.dataloaders.tasks_provider import TaskList, prepare_classes_list, RehearsalTask
from taskIL import train_validation_all_classes
from dataloaders.cifar10 import cifar10_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining
import numpy as np
import torch.optim as optim
import wandb
from utils import get_device

model_dict = {
    "SmallAlexNetTasklIL": SmallAlexNetTaslIL,
    "ResNet18": ResNet18IL,
}

# config_name = "AlexNetTaskILNoise"
config_name = "ResNetTaskILNoise"


@hydra.main(version_base=None, config_path="configs/experiments", config_name=config_name)
def main(cfg) -> None:
    device = get_device()

    print("Running experiment with settings:\n")
    print(OmegaConf.to_yaml(cfg))

    num_classes, classes_per_task = 10, 2
    classes_list = prepare_classes_list(num_classes, classes_per_task)
    tasks = TaskList(classes_list, cfg['batch_size'], torchvision.datasets.CIFAR10)
    tasks_test = TaskList(classes_list, cfg['batch_size'], torchvision.datasets.CIFAR10, train=False)

    if cfg['rehearsal_dataset']:
        rehearsal_loader = dataloader_pretraining(cfg['rehearsal_dataset'], no_classes=2, batch_size=cfg['batch_size_rehearsal'])
    else:
        rehearsal_loader = None

    wandb.init(
        project=cfg['project'],
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled"
    )

    model_reference = model_dict[cfg['architecture']]
    model = model_reference(out_dim=10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    # pretraining
    if cfg['pretraining']:
        print("Running pretraining")
        train_validation_all_classes(model, optimizer, RehearsalTask(rehearsal_loader), device, tasks_test=tasks_test,
                                    epoch=cfg['epochs'], log_interval=10)

    # CL
    print("Running CL")
    train_validation_all_classes(model=model, optimizer=optimizer, tasks=tasks, device=device, tasks_test=tasks_test, rehearsal_loader=rehearsal_loader, epoch=cfg['epochs'], log_interval=10)


if __name__ == "__main__":
    main()
