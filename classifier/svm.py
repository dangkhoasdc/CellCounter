"""
File: svm.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: SVM Classifier: use cross-validation and grid search
in order to find the best parameters
"""

class SVM(object):
    def __init__(self, k_fold=3, training_data=None, testing_data=None):
        self.k_fold = k_fold
        self.training_data = training_data
        self.testing_data = testing_data
        <`0`>
