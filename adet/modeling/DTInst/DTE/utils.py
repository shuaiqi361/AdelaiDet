# coding:utf-8

import numpy as np
import torch
import math
import cv2


class IOUMetric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def soft_thresholding(x, lm):
    ze_ = torch.zeros(size=x.size(), device=x.device)
    return torch.sign(x) * torch.maximum(torch.abs(x) - lm, ze_)


@torch.no_grad()
def fast_ista(b, A, lmbda, max_iter):
    """
    This is the fast Iterative Shrinkage-Thresholding Algorithm to solve the following objective:
    min: {L2_norm(Ax - b) + L1_norm(x)}
    :param b: input data with shape: [n_samples, n_features]
    :param A: a pre-learned Dictionary, with shape: [n_coeffs, n_features]
    :param lmbda: sparsity term to control the importance of the L1 term
    :param max_iter:
    :return: sparse codes with shape: [n_samples, n_coeffs]
    """
    n_coeffs, n_feats = A.size()
    n_samples = b.size()[0]
    x = torch.zeros(size=(n_samples, n_coeffs), device=b.device)
    t = 1.
    z = torch.zeros(size=(n_samples, n_coeffs), device=b.device)
    L = torch.linalg.norm(A, ord=2) ** 2  # Lipschitz constant, 2-norm (largest sing. value)

    for k in range(max_iter):
        x_old = x.clone()
        z = z + torch.matmul(b - torch.matmul(z, A), A.T) / L
        x = soft_thresholding(z, lmbda / L)
        t0 = t
        t = (1. + math.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - x_old)

    return x


def prepare_distance_transform_from_mask(masks, mask_size, kernel=3, dist_type=cv2.DIST_L2):
    """
    Given a set of masks as torch tensor, convert to numpy array, find distance transform maps from them,
    and convert DTMs back to torch tensor
    :param dist_type: used for distance transform
    :param kernel: kernel size for distance transforms
    :param masks: input masks for instance segmentation, shape: (N, mask_size, mask_size)
    :param mask_size: input mask size
    :return: a set of distance transform maps in torch tensor, with the same shape as input masks
    """
    assert mask_size * mask_size == masks.shape[1]
    device = masks.device
    masks = masks.view(masks.shape[0], mask_size, mask_size).cpu().numpy()
    masks = masks.astype(np.uint8)
    DTMs = []
    for m in masks:
        dist_m = cv2.distanceTransform(m, distanceType=dist_type, maskSize=kernel)
        dist_m = dist_m / np.max(dist_m)
        dist_map = np.where(dist_m > 0, dist_m, -1).astype(np.float32)  # DTM in (-1, 0-1)

        DTMs.append(dist_map.reshape((1, -1)))

    DTMs = np.concatenate(DTMs, axis=0)
    DTMs = torch.from_numpy(DTMs).to(torch.float32).to(device)

    return DTMs


def prepare_overlay_DTMs_from_mask(masks, mask_size, kernel=3, dist_type=cv2.DIST_L2):
    """
    Given a set of masks as torch tensor, convert to numpy array, find distance transform maps from them,
    and convert DTMs back to torch tensor
    :param dist_type: used for distance transform
    :param kernel: kernel size for distance transforms
    :param masks: input masks for instance segmentation, shape: (N, mask_size, mask_size)
    :param mask_size: input mask size
    :return: a set of distance transform maps in torch tensor, with the same shape as input masks
    """
    assert mask_size * mask_size == masks.shape[1]
    device = masks.device
    masks = masks.view(masks.shape[0], mask_size, mask_size).cpu().numpy()
    masks = masks.astype(np.uint8)
    DTMs = []
    for m in masks:
        dist_m = cv2.distanceTransform(m, distanceType=dist_type, maskSize=kernel)
        dist_m = dist_m / np.max(dist_m)
        dist_map = dist_m + m * 1.

        DTMs.append(dist_map.reshape((1, -1)))

    DTMs = np.concatenate(DTMs, axis=0)
    DTMs = torch.from_numpy(DTMs).to(torch.float32).to(device)

    return DTMs


def prepare_extended_DTMs_from_mask(masks, mask_size, kernel=3, dist_type=cv2.DIST_L2):
    """
    Given a set of masks as torch tensor, convert to numpy array, find distance transform maps from them,
    and convert DTMs back to torch tensor
    :param dist_type: used for distance transform
    :param kernel: kernel size for distance transforms
    :param masks: input masks for instance segmentation, shape: (N, mask_size, mask_size)
    :param mask_size: input mask size
    :return: a set of distance transform maps in torch tensor, with the same shape as input masks
    """
    assert mask_size * mask_size == masks.shape[1]
    device = masks.device
    masks = masks.view(masks.shape[0], mask_size, mask_size).cpu().numpy()
    masks = masks.astype(np.uint8)
    DTMs = []
    for m in masks:
        if np.sum(m) < 5:  # for foreground object
            dist_bbox_in = np.zeros(shape=m.shape)
        else:
            dist_bbox_in = cv2.distanceTransform(m, distanceType=cv2.DIST_L2, maskSize=3)
            dist_bbox_in = dist_bbox_in / np.max(dist_bbox_in)

        if np.sum(1 - m) < 5:  # for background
            dist_bbox_out = np.zeros(shape=m.shape)
        else:
            dist_bbox_out = cv2.distanceTransform(1 - m, distanceType=cv2.DIST_L2, maskSize=3)
            dist_bbox_out = -dist_bbox_out / np.max(dist_bbox_out)

        dist_map = (dist_bbox_in + dist_bbox_out + 1) / 2.

        DTMs.append(dist_map.reshape((1, -1)))

    DTMs = np.concatenate(DTMs, axis=0)
    DTMs = torch.from_numpy(DTMs).to(torch.float32).to(device)

    return DTMs
