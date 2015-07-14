"""
File: framework.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The proposed framework
"""

class Framework(object):
    """Main Framework"""
    def __init__(self, training_files=None, testing_files=None):
        """init"""
        self.training_data = training_files
        self.testing_data = testing_files
        self.preprocess_method = None
        self.extract_method = None

    def preprocess(self, method):
        self.preprocess_method = method

    def extract_features(self, method):
        self.extract_method = method
    def train(self):
        print "training phase"

