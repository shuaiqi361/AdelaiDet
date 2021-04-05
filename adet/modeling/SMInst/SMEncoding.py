# coding:utf-8
import torch
import torch.nn as nn
from .SME import fast_ista


@torch.no_grad()
class SparseMaskEncoding(nn.Module):
    """
    To do the sparse encoding for masks using Fast ISTA method.
        dictionary: (tensor), shape (n_coeffs, n_features)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_codes = cfg.MODEL.SMInst.NUM_CODE
        self.mask_size = cfg.MODEL.SMInst.MASK_SIZE
        self.sparse_alpha = cfg.MODEL.SMInst.MASK_SPARSE_ALPHA
        self.max_iter = cfg.MODEL.SMInst.MAX_ISTA_ITER
        self.dictionary = nn.Parameter(torch.zeros(self.num_codes, self.mask_size ** 2), requires_grad=False)

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

        X_transformed = fast_ista(X * 1., self.dictionary, lmbda=self.sparse_alpha, max_iter=self.max_iter)

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
            X_transformed_img = X_transformed > 0.5  # the predicted binary mask
            return X_transformed, X_transformed_img
        else:
            X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)

        return X_transformed
