# coding:utf-8
import torch
import torch.nn as nn
import cv2
from .SparseDTMEncoding import fast_ista, \
    prepare_distance_transform_from_mask_with_weights, \
    prepare_reciprocal_DTM_from_mask, prepare_complement_DTM_from_mask


@torch.no_grad()
class DistanceTransformEncoding(nn.Module):
    """
    To do the sparse encoding for masks using Fast ISTA method.
        dictionary: (tensor), shape (n_coeffs, n_features)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_codes = cfg.MODEL.DTInst.NUM_CODE
        self.mask_size = cfg.MODEL.DTInst.MASK_SIZE
        self.fg_weighting = cfg.MODEL.DTInst.FOREGROUND_WEIGHTING
        self.bg_weighting = cfg.MODEL.DTInst.BACKGROUND_WEIGHTING
        self.mask_bias = cfg.MODEL.DTInst.MASK_BIAS
        self.sparse_alpha = cfg.MODEL.DTInst.MASK_SPARSE_ALPHA
        self.max_iter = cfg.MODEL.DTInst.MAX_ISTA_ITER
        self.dictionary = nn.Parameter(torch.zeros(self.num_codes, self.mask_size ** 2), requires_grad=False)
        self.shape_mean = nn.Parameter(torch.zeros(1, self.mask_size ** 2), requires_grad=False)
        self.shape_std = nn.Parameter(torch.zeros(1, self.mask_size ** 2), requires_grad=False)
        self.if_whiten = cfg.MODEL.SMInst.WHITEN

        if cfg.MODEL.DTInst.DIST_TYPE == 'L2':
            self.dist_type = cv2.DIST_L2
        elif cfg.MODEL.DTInst.DIST_TYPE == 'L1':
            self.dist_type = cv2.DIST_L1
        elif cfg.MODEL.DTInst.DIST_TYPE == 'C':
            self.dist_type = cv2.DIST_C
        else:
            self.dist_type = cv2.DIST_L2

    def encoder(self, X):
        """
        Fast ISTA on X, encode X as a set of sparse codes.
        Parameters
        ----------
        X : Original features(tensor), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : Transformed features(tensor), shape (n_samples, num_codes)
        """
        assert X.shape[1] == self.mask_size ** 2, print("The original mask_size of input"
                                                        " should be equal to the supposed size.")

        # X_t, weight_maps, hd_maps = prepare_distance_transform_from_mask_with_weights(X, self.mask_size,
        #                                                                               dist_type=self.dist_type,
        #                                                                               fg_weighting=self.fg_weighting,
        #                                                                               bg_weighting=self.bg_weighting,
        #                                                                               mask_bias=self.mask_bias)
        X_t = prepare_complement_DTM_from_mask(X, self.mask_size, dist_type=self.dist_type)

        if self.if_whiten:
            Centered_X = (X_t - self.shape_mean) / self.shape_std
        else:
            Centered_X = X_t - self.shape_mean

        X_transformed = fast_ista(Centered_X, self.dictionary, lmbda=self.sparse_alpha, max_iter=self.max_iter)

        # return X_transformed, Centered_X, weight_maps, hd_maps
        return X_transformed

    def decoder(self, X, is_train=False):
        """
        Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Parameters
        ----------
        X : Encoded features(tensor), shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original original features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.num_codes, print("The dim of transformed data "
                                                   "should be equal to the supposed dim.")

        if self.if_whiten:
            X_transformed = torch.matmul(X, self.dictionary) * self.shape_std + self.shape_mean
        else:
            X_transformed = torch.matmul(X, self.dictionary) + self.shape_mean

        if is_train:
            # X_transformed_img = X_transformed + 0.9 >= 0.5  # the predicted binary mask for DTMs
            # X_transformed_img = X_transformed + 0.6 >= 0.5  # the predicted binary mask for reciprocal DTMs
            X_transformed_img = X_transformed + 0.55 >= 0.5  # the predicted binary mask for complement DTMs
            return X_transformed, X_transformed_img
        else:
            X_transformed = torch.clamp(X_transformed + 0.55, min=0.01, max=0.99)  # for normal DTM

        return X_transformed
