import torch
import numpy as np


def kl_divergence(rho, rho_hat):
    """
    :param rho: desired average activity, should be small
    :param rho_hat: network pre-act outputs, sigmoid is applied in this function
    :return:
    """
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)  # sigmoid because we need the probability distributions
    rho = torch.ones(size=rho_hat.size()) * rho
    rho = rho.to(rho_hat.device)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def huber(true, pred, delta, reduction="none"):
    loss = np.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                    delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
    if reduction == 'none':
        return loss
    else:
        return torch.sum(loss)


# log cosh loss
def logcosh(true, pred, reduction="none"):
    loss = torch.log(torch.cosh(pred - true))
    if reduction == 'none':
        return loss
    else:
        return torch.sum(loss)
