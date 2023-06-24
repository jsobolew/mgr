import torch
from Nets import SmallAlexNetTaslIL
from taskIL import train_validation_all_classes
from dataloaders.cifar10 import cifar10_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining_gray
import numpy as np
import torch.optim as optim
import wandb
import torch.nn.functional as F

def train_validation_all_classes(model, optimizer, tasks, device, epoch=1, log_interval = 1000):
    train_losses = []
    tasks_acc = [[], [], [], [], []]
    exemplers = []

    for taskNo in range(len(tasks)):
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(tasks[taskNo]):
                model.train()

                # training on task
                output = model(taskNo, data.to(device))
                loss = F.cross_entropy(output, target.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(f"Train epoch: {e} [{batch_idx * len(data)} / {len(tasks[taskNo].dataset)}]       loss: {loss.item()}")
                    acc_tasks = {}
                    for i in range(len(tasks)):
                        curr_task_acc = test(model, tasks[i], i, device, print_accuracy=False)
                        tasks_acc[i].append(curr_task_acc)
                        acc_tasks.update({f"acc_task_{i}": curr_task_acc})
                    wandb.log(acc_tasks)
                    print(acc_tasks)

                train_losses.append(loss.cpu().item())
                wandb.log({"loss": loss.item()})
                exemplars = tasks[taskNo].batch_size * batch_idx
                exemplers.append(exemplars)
                wandb.log({"exemplers": exemplers})
    return train_losses, tasks_acc, exemplers

def test(model, test_loader, taskNo, device, print_accuracy=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(taskNo, data.to(device))
            test_loss += F.cross_entropy(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    classes = np.arange(10)
    np.random.shuffle(classes)
    classes = list(classes)

    task1 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task2 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task3 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task4 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())
    task5 = cifar10_for_classes_TaskIL(classes.pop(), classes.pop())

    tasks = [task1 ,task2, task3, task4, task5]

    wandb.init(
        # set the wandb project where this run will be logged
        project="no rehersal Alexnet MNIST Task IL",
        
        # track hyperparameters and run metadata
        config={
        "setup": "task IL",
        "learning_rate": 0.1,
        "architecture": "SmallAlexNet",
        "dataset": "MNIST",
        "epochs": 5,
        },
        mode="disabled"
    )

    model = SmallAlexNetTaslIL(feat_dim=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_losses, tasks_losses, exemplers = train_validation_all_classes(model, optimizer, tasks, device, epoch=5, log_interval = 10)

if __name__ == "__main__":
    main()