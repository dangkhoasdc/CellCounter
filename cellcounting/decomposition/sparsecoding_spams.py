"""
File: sparsecoding.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Sparse Coding implementation using SPAMS lib
"""
import numpy as np
import spams


def encode(X, dictionary, alpha):
    """
    Sparse coding
    """
    X1 = np.asfortranarray(X.T)
    result = spams.omp(X1, dictionary, lambda1=alpha)
    return result.T.toarray()


class DictLearning:
    """
    Solves a dictionary learning matrix factorization problem.
    """
    def __init__(self, X, n_components, alpha, save=None):
        X1 = np.asfortranarray(X.T)
        self.X = X1
        self.lambda1 = alpha
        self.d = spams.trainDL(X1, K=n_components, mode=3, modeD=0, numThreads=-1, lambda1=alpha, return_model=False)
        self.d = np.asfortranarray(self.d)
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
        self.code = spams.omp(self.X1, self.d, lambda1=self.lambda1)
        return self.code.todense(out=np.ndarray)
