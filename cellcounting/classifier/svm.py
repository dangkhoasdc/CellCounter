"""
File: svm.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: SVM Classifier: use cross-validation and grid search
in order to find the best parameters
"""
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


class SVM(object):
    """ Wrapped SVM class from sklearn lib """
    default_c_range = np.logspace(-1, 4, num=8, base=5)
    default_gamma_range = np.logspace(-3, 1, num=8, base=15)

    def __init__(self, k_fold=3, training_data=None, training_labels=None,
                 testing_data=None, testing_labels=None):

        self.k_fold = k_fold
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels
        self._model = None
        self._best_params = None

    def load(self):
        """ load the model from file """
        self._model = joblib.load("model")

    def auto_train(self, samples=None, labels=None, k_fold=None, save=True,
                   c_range=None, gamma_range=None):
        """ The idea comes from auto_train method in OpenCV lib """

        if k_fold is not None:
            self.k_fold = k_fold
        if samples is not None and labels is not None:
            self.training_data = samples
            self.training_labels = labels

        assert type(self.training_data) is np.ndarray
        assert type(self.training_labels) is np.ndarray
        # grid search + cross-validation
        c_range = c_range if c_range is not None else self.default_c_range

        if gamma_range is None:
            gamma_range = self.default_gamma_range

        param_grid = {"C": c_range, "gamma": gamma_range}
        clf = SVC(kernel="rbf", tol=10e-6)
        print self.k_fold
        self._model = GridSearchCV(clf,
                                   param_grid=param_grid,
                                   n_jobs=-1,
                                   cv=self.k_fold)

        self._model.fit(self.training_data, self.training_labels)

        self._best_params = self._model.best_params_
        if save:
            joblib.dump(self._model, "model")

    def predict(self, sample):
        """ predict the label of a sample """
        return self._model.predict(sample)

    def __str__(self):
        return "SVM Model :" + str(self._best_params)
