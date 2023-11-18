import hydra
import numpy as np
import torchvision
from omegaconf import OmegaConf
from Nets import SmallAlexNetTaslIL, ResNet18IL, SmallAlexNet, MNIST_net, ResNet18
from dataloaders.tasks_provider import TaskList, prepare_classes_list, RehearsalTask
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
            "MNIST_net": MNIST_net,
            "SmallAlexNet": SmallAlexNet,
            "ResNet18": ResNet18,
        }
}

dataset_dict = {
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "MNIST": torchvision.datasets.MNIST,
}

# config_name = "AlexNetTaskILNoise"
config_name = "AlexNetTaskILNoiseCIFAR100"
# config_name = "AlexNetClassILNoise"


# config_name = "ResNetTaskILNoise"
# config_name = "MNISTNAPClassILNoise"


@hydra.main(version_base=None, config_path="configs/experiments", config_name=config_name)
def main(cfg) -> None:
    device = get_device()

    print("Running experiment with settings:\n")
    print(OmegaConf.to_yaml(cfg))

    classes_list = prepare_classes_list(cfg['num_classes'], cfg['classes_per_task'], cfg['dataset'])
    tasks = TaskList(classes_list, cfg['batch_size'], dataset_dict[cfg['dataset']], setup=cfg['setup'])
    tasks_test = TaskList(classes_list, cfg['batch_size'], dataset_dict[cfg['dataset']], train=False,
                          setup=cfg['setup'])

    if cfg['rehearsal_dataset']:
        no_classes = cfg['classes_per_task'] if cfg['setup'] == 'taskIL' else cfg['num_classes']
        rehearsal_loader = dataloader_pretraining(cfg['rehearsal_dataset'], no_classes=no_classes,
                                                  batch_size=cfg['batch_size_rehearsal'])
        print(f"Rehearsal: samples: {len(rehearsal_loader.dataset)} classes: {np.arange(no_classes)}")
    else:
        rehearsal_loader = None

    # logging data parameters
    for i, (task, task_test) in enumerate(zip(tasks.tasks, tasks_test.tasks)):
        print(
            f"Task     : {i} samples: {len(task.dataset)} global classes: {task.global_classes} local classes: {task.dataset.targets.unique()}")
        print(
            f"Task test: {i} samples: {len(task_test.dataset)} global classes: {task_test.global_classes} local classes: {task_test.dataset.targets.unique()}")

    config = OmegaConf.to_container(cfg, resolve=True)
    config['classes_list'] = classes_list
    wandb.init(
        project=cfg['project'],
        config=config,
        # mode="disabled"
    )

    model_reference = model_dict[cfg['setup']][cfg['architecture']]
    model = model_reference(out_dim=cfg['num_classes'], classes_per_task=cfg['classes_per_task']).to(device)

    if cfg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    elif cfg['optimizer'] == 'SGD_momentum':
        optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'], momentum=0.9)
    else: # SGD and any other value
        optimizer = optim.SGD(model.parameters(), lr=cfg['learning_rate'])

    # pretraining
    if cfg['pretraining']:
        print("Running pretraining")
        train_validation_all_classes(model=model, optimizer=optimizer, tasks=RehearsalTask(rehearsal_loader), device=device, tasks_test=None,
                                     epoch=1, log_interval=10)

    # CL
    print("Running CL")
    train_validation_all_classes(model=model, optimizer=optimizer, tasks=tasks, device=device, tasks_test=tasks_test,
                                 rehearsal_loader=rehearsal_loader, epoch=cfg['epochs'], log_interval=10,
                                 setup=cfg['setup'])


if __name__ == "__main__":
    main()
