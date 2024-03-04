import torch
import torch.nn.functional as F
import wandb

from loss import SupervisedContrastiveLoss
from utils import SafeIterator


def train_validation_all_classes(model, tasks, device, tasks_test=None, rehearsal_loader=None, epoch=1,
                                 log_interval=1000, setup='taskIL', contrastive_epoch=0, contrastive_optimizer=None):
    assert setup == 'taskIL' or setup == 'classIL', f"setup should be either taskIL or classIL but is {setup}"
    print(f"Starting training in {setup} setup")

    if rehearsal_loader:
        rehearsal_iter = SafeIterator(rehearsal_loader)

    for taskNo in range(len(tasks.tasks)):
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(tasks.tasks[taskNo].dataloader):

                # training on task
                if setup == 'taskIL':
                    output = model.forward(taskNo, data.numpy(), target)
                elif setup == 'classIL':
                    output = model(data.to(device))

                if rehearsal_loader and contrastive_epoch == 0:
                    # noise rehearsal
                    rehearsal_data = rehearsal_iter.next()
                    if setup == 'taskIL':
                        output = model(taskNo, rehearsal_data[0].to(device), target)
                    elif setup == 'classIL':
                        output = model(rehearsal_data[0].to(device))

                if batch_idx % log_interval == 0:
                    print(
                        f"Task: {taskNo} Train epoch: {e} [{batch_idx} / {len(tasks.tasks[taskNo].dataloader)}]       loss: {loss.item()}")
                    if tasks_test:
                        acc_tasks, acc_test_tasks = {}, {}
                        for i in range(len(tasks.tasks)):
                            curr_task_acc = test(model, tasks.tasks[i].dataloader, i, device, print_accuracy=False, setup=setup)
                            acc_tasks.update({f"acc_task_{i}": float(curr_task_acc)})

                            curr_test_task_acc = test(model, tasks_test.tasks[i].dataloader, i, device, print_accuracy=False, setup=setup)
                            acc_test_tasks.update({f"acc_test_task_{i}": float(curr_test_task_acc)})

                        wandb.log(acc_tasks)
                        wandb.log(acc_test_tasks)
                        print(acc_tasks)

                # wandb.log({"loss": loss.item()})


def test(model, test_loader, task_no, device, print_accuracy=True, setup='taskIL'):

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if setup == 'taskIL':
                output = model(task_no, data.to(device))
            elif setup == 'classIL':
                output = model(data.to(device))
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
    if print_accuracy:
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
