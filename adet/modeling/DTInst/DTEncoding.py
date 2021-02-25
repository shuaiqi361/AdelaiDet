# coding:utf-8
import torch
import torch.nn as nn
import cv2
from .DTE import fast_ista, prepare_distance_transform_from_mask


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
        self.sparse_alpha = cfg.MODEL.DTInst.MASK_SPARSE_ALPHA
        self.max_iter = cfg.MODEL.DTInst.MAX_ISTA_ITER
        self.dictionary = nn.Parameter(torch.zeros(self.num_codes, self.mask_size ** 2), requires_grad=False)
        if cfg.MODEL.DTInst.DIST_TYPE == 'L2':
            self.dist_type = cv2.DIST_L2
        elif cfg.MODEL.DTInst.DIST_TYPE == 'L1':
            self.dist_type = cv2.DIST_L1
        elif cfg.MODEL.DTInst.DIST_TYPE == 'C':
            self.dist_type = cv2.DIST_C
        elif cfg.MODEL.DTInst.DIST_TYPE == 'FAIR':
            self.dist_type = cv2.DIST_FAIR
        elif cfg.MODEL.DTInst.DIST_TYPE == 'WELSCH':
            self.dist_type = cv2.DIST_WELSCH
        elif cfg.MODEL.DTInst.DIST_TYPE == 'HUBER':
            self.dist_type = cv2.DIST_HUBER
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
        
        X_t = prepare_distance_transform_from_mask(X, self.mask_size, dist_type=self.dist_type)

        X_transformed = fast_ista(X_t, self.dictionary, lmbda=self.sparse_alpha, max_iter=self.max_iter)

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

        X_transformed = torch.matmul(X, self.dictionary)

        if is_train:
            pass
            # X_transformed = torch.clamp(X_transformed, min=0.001, max=0.999)
        else:
            X_transformed = torch.clamp(X_transformed + 1, min=0.001, max=0.999)

        return X_transformed
