"""
File: preprocess.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Stage Class Definition
"""
from .. import common as com
from ..segmentation.contour import Contour
from ..db import allidb
from sklearn.externals import joblib
import cv2
import numpy as np


class Framework(object):
    """Main Framework"""
    def __init__(self, window_size,
                 preprocess_stage,
                 segmentation_stage,
                 extraction,
                 classification):
        """init"""
        self._window_size = window_size
        self._preprocess = preprocess_stage
        self._segmentation = segmentation_stage
        self._classification = classification
        self._extraction = extraction
        self.preprocess_method = None
        self.extract_method = None
        self._model = None

    def preprocess(self, image):
        return self._preprocess.run(image)

    def extract_features(self, image):
        return self._extraction.run(image)

    def segment(self, image):
        return self._segmentation.run(image)

    def train(self, samples, labels, save):
        assert type(samples) is np.ndarray
        assert type(labels) is np.ndarray
        assert samples.shape[0] == labels.shape[1]
        model = self._classification.train_auto(samples, labels)
        self._model = model
        if save:
            joblib.dump(model, "model")
        return model

    def run_train(self, image_lst, loc_lst, save):
        assert self._window_size % 2 == 1
        assert type(image_lst) is list
        assert type(loc_lst) is list

        print "Training Phase"
        wd_sz = (self._window_size - 1) / 2
        # Preprocessing
        original_im = [cv2.imread(im, 1) for im in image_lst]
        images = [self.preprocess(im) for im in original_im]
        # Segmentation
        segments = [self.segment(im) for im in images]
        assert len(segments) != 0
        # Cropping from segmented image
        assert type(segments[0]) is Contour
        num_samples = len(com.flatten(segments))
        feature_size = len(self._extraction)
        training_samples = np.empty((num_samples, feature_size), dtype=float)
        training_labels = np.empty((1, num_samples), dtype=int)
        idx = 0
        for im_idx, (segs, cords) in enumerate(zip(segments, loc_lst)):
            for s in segs:
                value, point = com.nearest_point(s.center, cords)
                if value <= allidb.tol:
                    label = 1
                    cords.remove(value)
                else:
                    label = 0
                cropped_im = original_im[im_idx][s.center[0] - wd_sz: s.center[0] - wd_sz + 1,
                                                 s.center[1] - wd_sz: s.center[1] + wd_sz + 1]
                training_samples[idx] = self.extract_features(cropped_im)
                training_labels[idx] = label
                idx += 1
        self.train(training_samples, training_labels, save)

    def predict(self, image):
        """ predict if an image is a ALL cell or not """
        return self._model.predict(image)

    def run_test(self, image, loc_list):
        """ test an image """
        num_cells = 0
        correct_cells = 0
        wd_sz = (self._window_size - 1) / 2
        original_im = cv2.imread(image, 1)
        segments = self.segment(self.preprocess(original_im))
        for seg in segments:
            cropped_im = original_im[seg.center[0] - wd_sz: seg.center[0] + wd_sz + 1,
                                     seg.center[1] - wd_sz: seg.center[1] + wd_sz + 1]
            testing_sample = self.extract_features(cropped_im)

            result = self.predict(testing_sample)
            if result == 1:
                value = com.nearest_point(result, loc_list)
                if value <= allidb.tol:
                    correct_cells += 1
                num_cells += 1
        return num_cells


    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))