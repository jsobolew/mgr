import hydra
import numpy as np
import torchvision
from classification_models.models.resnet import ResNet18
from omegaconf import OmegaConf
from Nets import SmallAlexNetTaslIL, ResNet18IL, SmallAlexNet
from myCode.dataloaders.tasks_provider import TaskList, prepare_classes_list, RehearsalTask
from increamental_learning import train_validation_all_classes
from dataloaders.noise import dataloader_pretraining
import torch.optim as optim
import wandb
from utils import get_device

model_dict = {

    "taskIL":
        {
            "SmallAlexNet": SmallAlexNetTaslIL,
            "ResNet18": ResNet18IL,
        },
    "classIL":
        {
            "SmallAlexNet": SmallAlexNet,
            "ResNet18": ResNet18,
        }
}

# config_name = "AlexNetTaskILNoise"
config_name = "AlexNetClassILNoise"
# config_name = "ResNetTaskILNoise"


@hydra.main(version_base=None, config_path="configs/experiments", config_name=config_name)
def main(cfg) -> None:
    device = get_device()

    print("Running experiment with settings:\n")
    print(OmegaConf.to_yaml(cfg))

    num_classes, classes_per_task = 10, 2
    classes_list = prepare_classes_list(num_classes, classes_per_task)
    tasks = TaskList(classes_list, cfg['batch_size'], torchvision.datasets.CIFAR10, setup=cfg['setup'])
    tasks_test = TaskList(classes_list, cfg['batch_size'], torchvision.datasets.CIFAR10, train=False, setup=cfg['setup'])

    if cfg['rehearsal_dataset']:
        no_classes = classes_per_task if cfg['setup'] == 'taskIL' else num_classes
        rehearsal_loader = dataloader_pretraining(cfg['rehearsal_dataset'], no_classes=no_classes, batch_size=cfg['batch_size_rehearsal'])
        print(f"Rehearsal: samples: {len(rehearsal_loader.dataset)} classes: {np.arange(no_classes)}")
    else:
        rehearsal_loader = None

    # logging data parameters
    for i, (task, task_test) in enumerate(zip(tasks.tasks, tasks_test.tasks)):
        print(f"Task     : {i} samples: {len(task.dataset)} global classes: {task.global_classes} local classes: {task.dataset.targets.unique()}")
        print(f"Task test: {i} samples: {len(task_test.dataset)} global classes: {task_test.global_classes} local classes: {task_test.dataset.targets.unique()}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config['classes_list'] = classes_list
    wandb.init(
        project=cfg['project'],
        config=config,
        mode="disabled"
    )

    model_reference = model_dict[cfg['setup']][cfg['architecture']]
    model = model_reference(out_dim=num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    # pretraining
    if cfg['pretraining']:
        print("Running pretraining")
        train_validation_all_classes(model, optimizer, RehearsalTask(rehearsal_loader), device, tasks_test=tasks_test,
                                    epoch=cfg['epochs'], log_interval=10)

    # CL
    print("Running CL")
    train_validation_all_classes(model=model, optimizer=optimizer, tasks=tasks, device=device, tasks_test=tasks_test, rehearsal_loader=rehearsal_loader, epoch=cfg['epochs'], log_interval=10, setup=cfg['setup'])


if __name__ == "__main__":
    main()
