from dataclasses import dataclass

import numpy as np
import torch
import torchvision

from myCode.dataloaders.cifar10 import cifar10_for_classes_TaskIL


def prepare_classes_list(num_classes, classes_per_task) -> list:
    classes_per_task = classes_per_task
    num_tasks = num_classes // classes_per_task

    classes = np.arange(num_classes)
    np.random.shuffle(classes)
    classes = list(classes)
    classes = classes[:num_tasks * classes_per_task]

    classes_list = [[classes.pop() for _ in range(classes_per_task)] for _ in range(num_tasks)]
    return classes_list

@dataclass
class TaskList:
    def __init__(self, classes_list, batch_size, dataset, train=True, setup='taskIL'):
        self.classes_list = classes_list
        self.dataset = dataset
        self.train = train

        self.tasks = [Task(classes, batch_size, dataset, train, setup) for classes in self.classes_list]


class RehearsalTask:
    def __init__(self, rehearsal_loader):
        self.tasks = [rehearsal_loader]


class Task:
    def __init__(self, global_classes, batch_size, dataset_ref, train, setup='taskIL'):
        self.global_classes = global_classes
        self.batch_size_train = batch_size
        self.dataset_ref = dataset_ref
        self.setup = setup
        self.dataset = self.dataset_for_classes_task(self.dataset_ref, train)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def dataset_for_classes_task(self, dataset, train=True):
        dataset = dataset('files/', train=train, download=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              self.provide_normalization(dataset)
                          ]), )

        # select data
        dataset.data = np.concatenate([dataset.data[np.array(dataset.targets) == i] for i in self.global_classes])
        dataset.targets = torch.tensor(np.concatenate(
            [np.array(dataset.targets)[np.array(dataset.targets) == i] for i in self.global_classes]
        )).to(torch.int64)

        if self.setup == 'taskIL':
            # translate global to local labels
            for i in range(len(self.global_classes)):
                dataset.targets[np.array(dataset.targets) == self.global_classes[i]] = i

        return dataset

    @staticmethod
    def provide_normalization(dataset) -> torchvision.transforms.Normalize:
        if dataset is torchvision.datasets.CIFAR10:
            return torchvision.transforms.Normalize(
                                  (0.4924, 0.4739, 0.4198),
                                  (0.1891, 0.1865, 0.1885))
        elif dataset is torchvision.datasets.MNIST:
            return torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
        else:
            raise ValueError("incorrect dataset")
