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


def tensor_to_dtm(masks, mask_size, kernel=5, dist_type=cv2.DIST_L2):
    device = masks.device
    masks = masks.view(masks.shape[0], mask_size, mask_size).cpu().numpy()
    masks = masks.astype(np.uint8)
    DTMs = []

    for m in masks:
        dist_m = cv2.distanceTransform(m, distanceType=dist_type, maskSize=kernel)
        dist_m = dist_m / max(np.max(dist_m), 1.)  # basic dtms in (0, 1)
        dist_map = np.where(dist_m > 0, dist_m, -1).astype(np.float32)  # DTM in (-1, 0-1)
        DTMs.append(dist_map.reshape((1, -1)))

    DTMs = np.concatenate(DTMs, axis=0)
    DTMs = torch.from_numpy(DTMs).to(torch.float32).to(device)

    return DTMs


def prepare_distance_transform_from_mask(masks, mask_size, kernel=5, dist_type=cv2.DIST_L2):
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


def prepare_distance_transform_from_mask_with_weights(masks, mask_size, kernel=5, dist_type=cv2.DIST_L2, fg_weighting=1.0, bg_weighting=1.0, mask_bias=-0.1):
    """
    Given a set of masks as torch tensor, convert to numpy array, find distance transform maps from them,
    and convert DTMs back to torch tensor, a weight map with 1 - DTM will be returned(emphasizing boundary and thin parts)
    :param mask_bias: bias set for the pixels outside the contour
    :param fg_weighting: weighting for foreground pixels on the DTMs
    :param bg_weighting: weighting for background pixels on the DTMs
    :param dist_type: used for distance transform
    :param kernel: kernel size for distance transforms
    :param masks: input masks for instance segmentation, shape: (N, mask_size, mask_size)
    :param mask_size: input mask size
    :return: a set of distance transform maps, and a weight map in torch tensor, with the same shape as input masks
    """
    assert mask_size * mask_size == masks.shape[1]
    device = masks.device
    masks = masks.view(masks.shape[0], mask_size, mask_size).cpu().numpy()
    masks = masks.astype(np.uint8)
    DTMs = []
    weight_maps = []
    HD_maps = []
    for m in masks:
        dist_m = cv2.distanceTransform(m, distanceType=dist_type, maskSize=kernel)
        # dist_m_bg = cv2.distanceTransform(1 - m, distanceType=dist_type, maskSize=kernel)
        dist_m = dist_m / max(np.max(dist_m), 1.)  # basic dtms in (0, 1)
        # dist_m_bg = dist_m_bg / max(np.max(dist_m_bg), 1.)
        weight_map = np.where(dist_m > 0, fg_weighting + bg_weighting - dist_m, bg_weighting).astype(np.float32)
        dist_map = np.where(dist_m > 0, dist_m, -1).astype(np.float32)  # DTM in (-1, 0-1)
        hd_map = np.where(dist_m > 0, dist_m ** 2., mask_bias).astype(np.float32)  # not sure why the best
        # hd_map = np.where(dist_m > 0, dist_m ** 2., 0.01).astype(np.float32)
        # hd_map = np.where(dist_m > 0, dist_m ** 2, dist_m_bg ** 2 / 2.).astype(np.float32)
        weight_maps.append(weight_map.reshape((1, -1)))
        DTMs.append(dist_map.reshape((1, -1)))
        HD_maps.append(hd_map.reshape((1, -1)))

    DTMs = np.concatenate(DTMs, axis=0)
    weight_maps = np.concatenate(weight_maps, axis=0)
    HD_maps = np.concatenate(HD_maps, axis=0)
    DTMs = torch.from_numpy(DTMs).to(torch.float32).to(device)
    weight_maps = torch.from_numpy(weight_maps).to(torch.float32).to(device)
    HD_maps = torch.from_numpy(HD_maps).to(torch.float32).to(device)

    return DTMs, weight_maps, HD_maps


def prepare_augmented_distance_transform_from_mask(masks, mask_size, kernel=3, dist_type=cv2.DIST_L2):
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
        obj_contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_map = np.zeros((mask_size, mask_size), dtype=np.uint8)
        for c in obj_contours:
            polygon = c.reshape((-1, 2))
            cv2.drawContours(contour_map, [polygon.astype(np.int32)], contourIdx=-1,
                             color=1, thickness=1)

        dist_m = cv2.distanceTransform(m, distanceType=dist_type, maskSize=kernel)
        dist_m = dist_m / np.max(dist_m)
        dist_m = np.where(contour_map == 1, contour_map, dist_m)
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
