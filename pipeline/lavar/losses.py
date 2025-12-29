import torch

def poisson_nll(y_hat: torch.Tensor, y: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    # y_hat assumed >= 0 if you use softplus
    # Poisson NLL (up to constant): y_hat - y * log(y_hat)
    return (y_hat - y * torch.log(y_hat.clamp_min(eps))).mean()