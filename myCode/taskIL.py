import torch
import torch.nn.functional as F
import wandb


def train_validation_all_classes(model, optimizer, tasks, device, tasks_test=None, rehesrsal_loader=None, epoch=1,
                                 log_interval=1000):
    if tasks_test is None:
        tasks_test = tasks

    for taskNo in range(len(tasks)):
        if rehesrsal_loader:
            rehearsal_iter = iter(rehesrsal_loader)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(tasks[taskNo]):
                model.train()

                # training on task
                output = model(taskNo, data.to(device))
                loss = F.cross_entropy(output, target.to(device))

                if rehesrsal_loader:
                    # noise rehearsal
                    rehearsal_data = next(rehearsal_iter)
                    output = model(taskNo, rehearsal_data[0].to(device))
                    loss += F.cross_entropy(output, rehearsal_data[1].to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    acc_tasks = {}
                    acc_test_tasks = {}
                    for i in range(len(tasks)):
                        curr_task_acc = test(model, tasks[i], i, device, print_accuracy=False)
                        acc_tasks.update({f"acc_task_{i}": curr_task_acc})

                        curr_test_task_acc = test(model, tasks_test[i], i, device, print_accuracy=False)
                        acc_test_tasks.update({f"acc_test_task_{i}": curr_test_task_acc})

                    wandb.log(acc_tasks)
                    wandb.log(acc_test_tasks)
                    print(
                        f"Train epoch: {e} [{batch_idx * len(data)} / {len(tasks[taskNo].dataset)}]       loss: {loss.item()}")
                    print(acc_tasks)

                wandb.log({"loss": loss.item()})


def test(model, test_loader, task_no, device, print_accuracy=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(task_no, data.to(device))
            test_loss += F.cross_entropy(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
