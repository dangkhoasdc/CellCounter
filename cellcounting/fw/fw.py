"""
File: preprocess.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Stage Class Definition
"""
import cv2
import numpy as np
from cellcounting.decomposition import sparsecoding_spams as sparse
from cellcounting.pooling import pooling
import cellcounting.common as com
from cellcounting.fw.absframework import AbsFramework
from cellcounting.fw.learning import LearningFramework


class Framework(LearningFramework):
    """Main Framework"""
    def __init__(self, database,
                 preprocess_stage,
                 segmentation_stage,
                 extraction,
                 classifier):
        """init"""
        super(Framework, self).__init__(database,
                                        preprocess_stage,
                                        segmentation_stage,
                                        extraction,
                                        classifier)

    def train(self, image_lst, loc_lst, save):
        """
        The training phase
        """
        # get the training data

        data, l, _ = self.get_data(image_lst, loc_lst)
        labels = []
        print "Get data done"
        training_data = []
        features = []
        for idx, im in enumerate(data):
            f = self._extraction.compute(im, self._extraction.detect(im))[1]
            if f is not None:
                features.append(f)
                labels.append(l[idx])
        learning_data = np.vstack(features)
        # sparse code
        self.alpha = 10
        sparsecode = sparse.DictLearning(learning_data, 300, self.alpha, save)
        self.dictionary = sparsecode.dictionary
        print "Build the dictionary done"
        # pooling
        for feature in features:
            coeffs = sparse.encode(feature, self.dictionary, self.alpha)
            vector = pooling.max_pooling(coeffs)
            training_data.append(vector)
        print "Max pooling function done"
        training_data = np.vstack(training_data)
        labels = np.float32(labels)

        print "The number of positive samples:", len(labels[labels==1])
        print "The number of false samples:", len(labels[labels==0])

        # svm
        print training_data.shape
        self._classifier.auto_train(training_data, labels)
        print "train done"
        return self._classifier


    def test(self, image, loc_lst, viz=False):
        """ test an image """
        demo = self.imread(image)
        locations = loc_lst
        data, l, segments = self.get_data([image], [loc_lst])
        num_samples = len(data)
        labels = []
        features = []

        for idx, im in enumerate(data):
            f = self._extraction.compute(im, self._extraction.detect(im))[1]
            if f is not None:
                features.append(f)
                labels.append(l[idx])

        testing_data = []

        for feature in features:
            coeffs = sparse.encode(feature, self.dictionary, self.alpha)
            vector = pooling.max_pooling(coeffs)
            testing_data.append(vector)

        testing_data = np.vstack(testing_data)
        testing_data = np.float32(testing_data)

        labels = np.float32(labels)
        result = self._classifier.predict(testing_data)

        for predicted, expected, s in zip(result, labels, segments):
            if predicted == expected:
                s.detected = True
        correct = len(filter(lambda x: x.detected, segments))
        if viz:
            com.visualize_segments(demo, segments, locations)
            com.debug_im(demo)
        return correct, total_segments


    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
