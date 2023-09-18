import hydra
from omegaconf import OmegaConf
from Nets import SmallAlexNetTaslIL
from taskIL import train_validation_all_classes
from dataloaders.cifar10 import cifar10_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining
import numpy as np
import torch.optim as optim
import wandb
from utils import get_device

model_dict = {
    "SmallAlexNetTasklIL": SmallAlexNetTaslIL
}


@hydra.main(version_base=None, config_path="configs/experiments", config_name="AlexNetTaskILNoise")
def main(cfg) -> None:
    device = get_device()

    print("Running experiment with settings:\n")
    print(OmegaConf.to_yaml(cfg))

    # ------
    classes = np.arange(10)
    np.random.shuffle(classes)
    classes_test = list(classes.copy())
    classes = list(classes)

    task1 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop(), batch_size_train=32)
    task2 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop(), batch_size_train=32)
    task3 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop(), batch_size_train=32)
    task4 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop(), batch_size_train=32)
    task5 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop(), batch_size_train=32)

    tasks = [task1, task2, task3, task4, task5]

    task1_test = cifar10_for_classes_TaskIL(classes_test.pop(), classes_test.pop(), train=False, batch_size_train=32)
    task2_test = cifar10_for_classes_TaskIL(classes_test.pop(), classes_test.pop(), train=False, batch_size_train=32)
    task3_test = cifar10_for_classes_TaskIL(classes_test.pop(), classes_test.pop(), train=False, batch_size_train=32)
    task4_test = cifar10_for_classes_TaskIL(classes_test.pop(), classes_test.pop(), train=False, batch_size_train=32)
    task5_test = cifar10_for_classes_TaskIL(classes_test.pop(), classes_test.pop(), train=False, batch_size_train=32)

    tasks_test = [task1_test, task2_test, task3_test, task4_test, task5_test]
    # ------


    if cfg['rehearsal_dataset']:
        rehearsal_loader = dataloader_pretraining(cfg['rehearsal_dataset'], no_classes=2)
    else:
        rehearsal_loader = None

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
            "rehearsal_dataset": cfg['rehearsal_dataset'],
        },
        # mode="disabled"
    )

    model_reference = model_dict[cfg['architecture']]
    model = model_reference(feat_dim=10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    train_validation_all_classes(model, optimizer, tasks, device, tasks_test=tasks_test, rehesrsal_loader=rehearsal_loader, epoch=cfg['epochs'], log_interval=10)


if __name__ == "__main__":
    main()
