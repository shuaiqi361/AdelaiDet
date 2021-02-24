# coding:utf-8

import numpy as np
import torch
import math
import cv2
from pycocotools import mask as cocomask


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


def uniformsample(pgtnp_px2, newpnum):
    """
    uniformly draw sample vertices from a given contour
    :param pgtnp_px2: original contour vertex coordinates
    :param newpnum: desired number of vertices
    :return: a new contour with desired number of vertices
    """
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.floor(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:
            if edgenumsum > newpnum:
                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def prepare_polygon_from_mask(masks, mask_size, num_vertex, pads=5):
    """
    Given a set of masks as torch tensor, convert to numpy array, find contours from it, resample vertices from masks,
    and convert back to torch tensor
    :param pads: initial kernel size for morphological closing operation
    :param num_vertex: number of vertex for polygons
    :param masks: input masks for instance segmentation, shape: (N, mask_size, mask_size)
    :param mask_size: input mask size
    :return: a set of polygon coordinates in torch tensor, shape: (N, num_vertex * 2)
    """
    assert mask_size * mask_size == masks.shape[1]
    masks = masks.view(masks.shape[0], mask_size, mask_size).numpy() * 255
    masks = np.where(masks >= 0.5 * 255, 1, 0).astype(np.uint8)
    polys = []
    for m in masks:
        obj_contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        while len(obj_contours) > 1:
            kernel = np.ones((pads, pads), np.uint8)  # apply closing for disconnected parts
            m_closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(m_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(obj_contours) > 1:
                if pads <= mask_size // 2:
                    pads += 5
                else:  # if kernel size exceeds a threshold, use the largest part
                    obj_contours = sorted(obj_contours, key=cv2.contourArea)  # get the largest masks
                    break

        contour = obj_contours[-1].reshape(-1, 2)
        uni_contour = uniformsample(contour, num_vertex)

        # normalize the shape and store in tensor
        norm_shape = uni_contour * 1. / mask_size
        polys.append(norm_shape.reshape((1, -1)))

    polys = np.concatenate(polys, axis=0)
    polys = torch.from_numpy(polys).to(torch.float32)

    return polys


def polygon_to_bitmask(polygons, height, width):
    """
    convert single polygon to mask
    :param polygons: List[numpy.array]
    :param height:
    :param width:
    :return: mask of type np.uint8
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = cocomask.frPyObjects(polygons, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle).astype(np.uint8)  # binary masks


def poly_to_mask(polys, mask_size):
    """
    convert polygon to a squared mask
    :param polys: is a list of polygons(torch tensor), it could have multiple disconnected parts
    :param mask_size: int, the size of the squared masks
    :return: a binary mask with
    """
    device = polys.device
    polys = polys.cpu().numpy()
    polys = np.floor(np.clip(polys, 0, 0.999) * mask_size).astype(np.int32)  # coordinates in masks
    masks = [polygon_to_bitmask([p], mask_size, mask_size) for p in polys]
    masks = np.concatenate(masks, axis=0).reshape(-1, mask_size * mask_size)

    return torch.from_numpy(masks).to(device)
