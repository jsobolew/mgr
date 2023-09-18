import numpy as np
import torch
import torchvision

from myCode.dataloaders.cifar10 import cifar10_for_classes_TaskIL
torchvision.datasets.CIFAR10

class TaskList:
    def __init__(self, num_classes, classes_per_task, dataset):
        self.classes_per_task = classes_per_task
        self.num_tasks = num_classes // classes_per_task

        self.classes = np.arange(num_classes)
        np.random.shuffle(self.classes)
        self.classes = list(self.classes)
        self.classes = self.classes[:self.num_tasks*self.classes_per_task]

        self.classes_list = [[self.classes.pop() for _ in range(classes_per_task)] for _ in range(self.num_tasks)]

        self.tasks = [TaskIL(classes, dataset) for classes in self.classes_list]


class TaskIL:
    def __init__(self, global_classes, dataset):
        self.global_classes = global_classes
        self.dataloader = dataset_for_classes_TaskIL(self.global_classes, dataset)



def dataset_for_classes_TaskIL(global_classes, dataset, batch_size_train=128, train=True):
    dataset = dataset('files/', train=train, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                    (0.4924, 0.4739, 0.4198),
                                    (0.1891, 0.1865, 0.1885))
                                ]),)

    dataset.data = np.concatenate(
        (dataset.data[np.array(dataset.targets) == class_1], dataset.data[np.array(dataset.targets) == class_2])
    )
    dataset.targets = torch.tensor(np.concatenate(
        (np.array(dataset.targets)[np.array(dataset.targets) == class_1], np.array(dataset.targets)[np.array(dataset.targets) == class_2])
    )).to(torch.int64)

    dataset.targets[np.array(dataset.targets) == class_1] = 0
    dataset.targets[np.array(dataset.targets) == class_2] = 1
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
    return train_loader
