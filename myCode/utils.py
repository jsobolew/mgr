import torch


def extract_mean_and_std(dataloader) -> torch.Tensor:
    mu, var = torch.tensor([]), torch.tensor([])

    for sample, _ in dataloader:
        mu = torch.cat((mu, torch.mean(sample, [2, 3])))
        var = torch.cat((var, torch.std(sample, [2, 3])))

    mean_mu = torch.mean(mu, 0)
    mean_var = torch.mean(var, 0)
    return mean_mu, mean_var

def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'