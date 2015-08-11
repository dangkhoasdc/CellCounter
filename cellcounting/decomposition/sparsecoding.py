"""
File: sparsecoding.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Sparse Coding implementation
"""
import numpy as np
import sklearn.decomposition as decomp


def encode(X, dictionary):
    """
    Sparse coding
    """
    return decomp.sparse_encode(X, dictionary)

class DictLearning:
    """
    Solves a dictionary learning matrix factorization problem.
    """
    def __init__(self, X, n_components, alpha, save=None):
        self.code, self.d = decomp.dict_learning(X,
                                                 n_components,
                                                 alpha)
        if save:
            np.save(save, self.d)

    @property
    def dictionary(self):
        """
        The dictionary factor in the matrix factorization
        """
        return self.d

    @property
    def coeffs(self):
        """
        The sparse code factor in the matrix factorization
        """
        return self.code


