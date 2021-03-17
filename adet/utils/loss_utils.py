import torch
import torch.nn.functional as F
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


def huber(pred, true, delta, reduction="none"):
    loss = np.where(torch.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                    delta * torch.abs(true - pred) - 0.5 * (delta ** 2))
    if reduction == 'none':
        return loss
    else:
        return torch.sum(loss)


# log cosh loss
def logcosh(pred, true, reduction="none"):
    loss = torch.log(torch.cosh(pred - true))
    if reduction == 'none':
        return loss
    else:
        return torch.sum(loss)


def loss_cos_sim(pred, true):
    # minimize average cosine similarity
    # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    # output = cos(true, pred)
    output = F.cosine_similarity(true, pred, dim=-1, eps=1e-6)
    return 1 - output


def loss_kl_div(pred, true):
    # output = F.kl_div(torch.sigmoid(true), torch.sigmoid(pred))
    # rho = torch.sigmoid(true)
    # rho_hat = torch.sigmoid(pred)
    # kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    input_log_softmax = F.log_softmax(pred, dim=1)
    target_softmax = F.softmax(true, dim=1)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')

    return kl_div


def smooth_l1_loss(pred, true):
    loss = F.smooth_l1_loss(pred, true, reduction='none')
    return loss
