"""
File: svm.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: SVM Classifier: use cross-validation and grid search
in order to find the best parameters
"""


class SVM(object):
    """ Wrapped SVM class from sklearn lib """
    def __init__(self, k_fold=3, training_data=None, training_labels=None,
                 testing_data=None, testing_labels=None):
        self.k_fold = k_fold
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def auto_train(self, samples=None, labels=None, k_fold=None):
        """ The idea comes from auto_train method in OpenCV lib """

        if k_fold is not None:
            self.k_fold = k_fold
        if samples is not None and labels is not None:
            self.training_data = samples
            self.training_labels = labels

        assert self.training_data is not None
        assert self.training_labels is not None
