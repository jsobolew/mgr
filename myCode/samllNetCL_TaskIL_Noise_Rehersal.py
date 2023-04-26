import torch
from Nets import NetTaskIL
from taskIL import train_validation_all_classes
from dataloaders.mnist import MNIST_for_classes_TaskIL
from dataloaders.noise import dataloader_pretraining_gray
import numpy as np
import torch.optim as optim
import wandb
import torch.nn.functional as F

def train_validation_all_classes(model, optimizer, tasks, rehersal_loader, epoch=1, log_interval = 1000):
    train_losses = []
    tasks_acc = [[], [], [], [], []]
    exemplers = []

    rehersal_iterator = iter(rehersal_loader)

    model.train()
    for taskNo in range(len(tasks)):
        for batch_idx, (data, target) in enumerate(tasks[taskNo]):

            # training on task
            optimizer.zero_grad()
            output = model(taskNo, data)
            loss = F.cross_entropy(output, target)

            # noise rehersal
            rehersal_data = next(rehersal_iterator)
            output = model(taskNo, rehersal_data[0])
            loss += F.cross_entropy(output, rehersal_data[1])
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(f"Train [{batch_idx * len(data)} / {len(tasks[taskNo].dataset)}]       loss: {loss.item()}")
                for i in range(len(tasks)):
                    curr_task_acc = test(model, tasks[i], i, print_accuracy=False)
                    tasks_acc[i].append(curr_task_acc)
                    wandb.log({f"acc_task_{i}": curr_task_acc})

            train_losses.append(loss.item())
            wandb.log({"loss": loss.item()})
            exemplars = tasks[taskNo].batch_size * batch_idx
            exemplers.append(exemplars)
            wandb.log({"exemplers": exemplers*2})
    return train_losses, tasks_acc, exemplers

def test(model, test_loader, taskNo, print_accuracy=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(taskNo, data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
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

    task1 = MNIST_for_classes_TaskIL(classes.pop(), classes.pop())
    task2 = MNIST_for_classes_TaskIL(classes.pop(), classes.pop())
    task3 = MNIST_for_classes_TaskIL(classes.pop(), classes.pop())
    task4 = MNIST_for_classes_TaskIL(classes.pop(), classes.pop())
    task5 = MNIST_for_classes_TaskIL(classes.pop(), classes.pop())

    tasks = [task1 ,task2, task3, task4, task5]

    rehersal_loader = dataloader_pretraining_gray("dead_leaves-squares", no_classes=2)


    wandb.init(
        # set the wandb project where this run will be logged
        project="rehersal small net MNIST Task IL",
        
        # track hyperparameters and run metadata
        config={
        "setup": "task IL",
        "learning_rate": 0.1,
        "architecture": "CNN",
        "dataset": "MNIST",
        "epochs": 1,
        }
    )

    model = NetTaskIL(10)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_losses, tasks_losses, exemplers = train_validation_all_classes(model, optimizer, tasks, rehersal_loader, epoch=1, log_interval = 10)

if __name__ == "__main__":
    main()