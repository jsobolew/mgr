import torch
import torch.nn.functional as F
import wandb


def train_validation_all_classes(model, optimizer, tasks, device, tasks_test=None, rehearsal_loader=None, epoch=1,
                                 log_interval=1000, setup='taskIL'):
    assert setup == 'taskIL' or setup == 'classIL', f"setup should be either taskIL or classIL but is {setup}"
    print(f"Starting training in {setup} setup")

    if tasks_test is None:
        tasks_test = tasks

    for taskNo in range(len(tasks.tasks)):
        if rehearsal_loader:
            rehearsal_iter = iter(rehearsal_loader)
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(tasks.tasks[taskNo].dataloader):
                model.train()

                # training on task
                if setup == 'taskIL':
                    output = model(taskNo, data.to(device))
                elif setup == 'classIL':
                    output = model(data.to(device))
                loss = F.cross_entropy(output, target.to(device))

                if rehearsal_loader:
                    # noise rehearsal
                    rehearsal_data = next(rehearsal_iter)
                    if setup == 'taskIL':
                        output = model(taskNo, rehearsal_data[0].to(device))
                    elif setup == 'classIL':
                        output = model(rehearsal_data[0].to(device))
                    loss += F.cross_entropy(output, rehearsal_data[1].to(device))
                    print(rehearsal_data[1])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    acc_tasks, acc_test_tasks = {}, {}
                    for i in range(len(tasks.tasks)):
                        curr_task_acc = test(model, tasks.tasks[i].dataloader, i, device, print_accuracy=False, setup=setup)
                        acc_tasks.update({f"acc_task_{i}": float(curr_task_acc)})

                        curr_test_task_acc = test(model, tasks_test.tasks[i].dataloader, i, device, print_accuracy=False, setup=setup)
                        acc_test_tasks.update({f"acc_test_task_{i}": float(curr_test_task_acc)})

                    wandb.log(acc_tasks)
                    wandb.log(acc_test_tasks)
                    print(
                        f"Train epoch: {e} [{batch_idx * len(data)} / {len(tasks.tasks[taskNo].dataloader)}]       loss: {loss.item()}")
                    print(acc_tasks)

                wandb.log({"loss": loss.item()})
        # if nap: # todo
        #     for x, y in dataloader:
        #         activations_sum, n_activation = extract_activations(x=x,model_feature_extractor=model_feature_extractor)


def test(model, test_loader, task_no, device, print_accuracy=True, setup='taskIL'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if setup == 'taskIL':
                output = model(task_no, data.to(device))
            elif setup == 'classIL':
                output = model(data.to(device))
            test_loss += F.cross_entropy(output, target.to(device), size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    if print_accuracy:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
