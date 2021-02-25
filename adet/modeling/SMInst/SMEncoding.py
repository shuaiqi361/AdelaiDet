# coding:utf-8
import torch
import torch.nn as nn
from .SME import fast_ista

VALUE_MAX = 0.05
VALUE_MIN = 0.01


@torch.no_grad()
class PCAMaskEncoding(nn.Module):
    """
    To do the mask encoding of PCA.
        components_: (tensor), shape (n_components, n_features) if agnostic=True
                                else (n_samples, n_components, n_features)
        explained_variance_: Variance explained by each of the selected components.
                            (tensor), shape (n_components) if agnostic=True
                                        else (n_samples, n_components)
        mean_: (tensor), shape (n_features) if agnostic=True
                          else (n_samples, n_features)
        agnostic: (bool), whether class_agnostic or class_specific.
        whiten : (bool), optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
        sigmoid: (bool) whether to apply inverse sigmoid before transform.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.agnostic = cfg.MODEL.MEInst.AGNOSTIC
        self.whiten = cfg.MODEL.MEInst.WHITEN
        self.sigmoid = cfg.MODEL.MEInst.SIGMOID
        self.dim_mask = cfg.MODEL.MEInst.DIM_MASK
        self.mask_size = cfg.MODEL.MEInst.MASK_SIZE

        if self.agnostic:
            self.components = nn.Parameter(torch.zeros(self.dim_mask, self.mask_size**2), requires_grad=False)
            self.explained_variances = nn.Parameter(torch.zeros(self.dim_mask), requires_grad=False)
            self.means = nn.Parameter(torch.zeros(self.mask_size**2), requires_grad=False)
        else:
            raise NotImplementedError

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        # In case of overflow
        value_random = VALUE_MAX * torch.rand_like(x)  # [0, 1) uniform distribution --> [0, 0.05)
        value_random = torch.where(value_random > VALUE_MIN, value_random, VALUE_MIN * torch.ones_like(x))  # [0.01, 0.05) uniform distribution
        x = torch.where(x > value_random, 1 - value_random, value_random)  # Near binary image, ([0.01, 0.05), (0.95, 0.99])
        # inverse sigmoid
        y = -1 * torch.log((1 - x) / x)
        return y

    def encoder(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : Original features(tensor), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : Transformed features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.mask_size**2, print("The original mask_size of input"
                                                      " should be equal to the supposed size.")

        if self.sigmoid:
            X = self.inverse_sigmoid(X)

        if self.agnostic:
            if self.means is not None:
                X_transformed = X - self.means
            X_transformed = torch.matmul(X_transformed, self.components.T)
            if self.whiten:
                X_transformed /= torch.sqrt(self.explained_variances)
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        return X_transformed

    def decoder(self, X, is_train=False):
        """Transform data back to its original space.
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
        assert X.shape[1] == self.dim_mask, print("The dim of transformed data "
                                                  "should be equal to the supposed dim.")

        if self.agnostic:
            if self.whiten:
                components_ = self.components * torch.sqrt(self.explained_variances.unsqueeze(1))
            X_transformed = torch.matmul(X, components_)
            if self.means is not None:
                X_transformed = X_transformed + self.means
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        if is_train:
            pass
        else:
            if self.sigmoid:
                X_transformed = torch.sigmoid(X_transformed)
            else:
                X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)

        return X_transformed


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
            pass
        else:
            X_transformed = torch.where(X_transformed >= 0.5, 1, 0)

        return X_transformed
