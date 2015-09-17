"""
File: globalfeatures.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The framework for the global feature
"""
import cv2
import numpy as np
import cellcounting.common as com
from cellcounting.fw.learning import LearningFramework


class Framework(LearningFramework):
    """Main Framework"""
    def __init__(self,
                 database,
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
        self._classifier = classifier
        self._extraction = extraction


    def train(self, image_lst, loc_lst, viz=False):
        """
        The training phase
        """
        # get the training data

        data, labels, _ = self.get_data(image_lst, loc_lst, viz)
        print "Get data done"
        training_data = [self._extraction.compute(im)[0] for im in data]
        training_data = np.vstack(training_data)
        labels = np.float32(labels)

        print "The number of positive samples:", len(labels[labels == 1])
        print "The number of false samples:", len(labels[labels == 0])

        # svm
        print training_data.shape
        self._classifier.auto_train(training_data, labels)
        print "train done"
        return self._classifier


    def test(self, image, loc_lst, viz=False):
        """ test an image """
        demo = self.imread(image)
        correct = 0
        locations = list(loc_lst)

        data, labels, segments = self.get_data([image], [loc_lst])
        total_segments = len(data)
        testing_data = [self._extraction.compute(im)[0] for im in data]

        testing_data = np.vstack(testing_data)

        result = self._classifier.predict(testing_data)

        for predicted, expected, s in zip(result, labels, segments):
            if predicted == expected:
                correct += 1
                s.detected = True
        if viz:
            self.visualize_segments(demo, segments, locations)
            com.debug_im(demo)
        return correct, total_segments

    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
