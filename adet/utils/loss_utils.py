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


def loss_cos_sim(true, pred):
    # minimize average cosine similarity
    # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    # output = cos(true, pred)
    output = F.cosine_similarity(true, pred, dim=-1, eps=1e-6)
    return output


def loss_kl_div(true, pred):
    # output = F.kl_div(torch.sigmoid(true), torch.sigmoid(pred))
    # rho = torch.sigmoid(true)
    # rho_hat = torch.sigmoid(pred)
    # kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    input_log_softmax = F.log_softmax(pred, dim=1)
    target_softmax = F.softmax(true, dim=1)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')

    return kl_div


def weighted_mse_loss(pred, target, weights):
    weights_ = weights + 1e-3
    mse_loss = F.mse_loss(pred, target, reduction='none')
    loss = torch.sum(mse_loss * weights_, dim=-1) / torch.sum(weights_, dim=-1)
    return loss
