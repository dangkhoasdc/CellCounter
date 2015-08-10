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
import cv2
import numpy as np
from .absframework import AbsFramework
from sklearn.preprocessing import normalize


class Framework(AbsFramework):
    """Main Framework"""
    def __init__(self, window_size,
                 scale_ratio,
                 preprocess_stage,
                 segmentation_stage,
                 extraction,
                 classification):
        """init"""
        super(Framework, self).__init__()
        self.scale_ratio = scale_ratio
        self._window_size = window_size
        self._preprocess = preprocess_stage
        self._segmentation = segmentation_stage
        self._classification = classification
        self._extraction = extraction
        self._model = None

    def preprocess(self, image):
        return self._preprocess.run(image)

    def extract(self, image):
        return self._extraction.run(image)

    def segment(self, image, demo):
        return self._segmentation.run(image, demo)

    def imread(self, fname, flags=1):
        """ load an image and scale it"""
        im = cv2.imread(fname, flags)
        height, width = im.shape[:2]

        height = int(self.scale_ratio * height)
        width  = int(self.scale_ratio * width)

        if im is None:
            raise IOError("Could not load an image ", fname)
        im = cv2.resize(im, (width, height))
        return im

    def get_data(self, image_lst, loc_lst, visualize=False):
        """
        prepare data for training phase
        """
        data = []
        labels = []
        for image, locs in zip(image_lst, loc_lst):

            demo_img = self.imread(image, 1)
            processed_img, gray_img = self.preprocess(demo_img)
            segments = self.segment(processed_img, gray_img, demo_img)

            correct = 0
            # draw all counted objects in the image

            if visualize:
                for seg in segments:
                    seg.draw(demo_img, (0, 255, 0), 1)
                for loc in locs:
                    cv2.circle(demo_img, loc, 2, (0, 255, 0), 1)

            # if there are more than 1 segment in this image
            if locs:
                # visualize true cells
                # check if each segment is close to one true cell
                for seg in segments:
                    data.append(seg)

                    if len(locs) == 0:
                        break

                    point, value = com.nearest_point(seg.center, locs)

                    if value <= self._db.tol:
                        locs.remove(point)
                        correct += 1
                        labels.append(1)

                        if visualize:
                            seg.draw(demo_img, (255, 255, 0), 1)
                    else:
                        labels.append(0)
            else:
                for seg in segments:
                    data.append(seg)
                    labels.append(0)

            if visualize:
                com.debug_im(gray_img)
                com.debug_im(processed_img)
                com.debug_im(demo_img, True)

        return data, labels
    def run_train(self, image_lst, loc_lst, save):
        assert self._window_size % 2 == 1
        assert type(image_lst) is list
        assert type(loc_lst) is list

        print "Training Phase"
        wd_sz = (self._window_size - 1) / 2
        # Preprocessing
        original_im = [self.imread(im, 1) for im in image_lst]
        images = [self.preprocess(im) for im in original_im]
        # Segmentation
        segments = [self.segment(im, demo_img) for im, demo_img in zip(images, original_im)]
        im_width = original_im[0].shape[1]
        im_height = original_im[0].shape[0]
        assert len(segments) != 0
        # Cropping from segmented image
        assert type(segments[0][0]) is Contour

        new_segments = []
        for seg in segments:
            seg = [s for s in seg if s.center[0] -wd_sz >= 0 and s.center[0] + wd_sz +1 < im_height]
            seg = [s for s in seg if s.center[1] -wd_sz >= 0 and s.center[1] + wd_sz +1 < im_width]
            new_segments.append(seg)
        segments = new_segments

        num_samples = len(com.flatten(segments))
        feature_size = len(self._extraction)
        training_samples = np.empty((num_samples, feature_size), dtype=float)
        training_labels = np.empty(num_samples, dtype=int)
        idx = 0
        for im_idx, (segs, cords) in enumerate(zip(segments, loc_lst)):
            for s in segs:
                point, value = com.nearest_point(s.center, cords)

                if value <= allidb.tol:
                    label = 1
                    cords.remove(point)
                else:
                    label = 0

                cropped_im = original_im[im_idx][s.center[0] - wd_sz: s.center[0] + wd_sz + 1,
                                                 s.center[1] - wd_sz: s.center[1] + wd_sz + 1]
                training_samples[idx] = self.extract(cropped_im)
                training_labels[idx] = label
                idx += 1

        assert type(training_samples) is np.ndarray
        assert type(training_labels) is np.ndarray
        assert training_samples.shape[0] == training_labels.shape[0]
        print training_labels
        normalize(training_samples, copy=False)
        self._classification.auto_train(training_samples, training_labels, save=save)

    def test(self, image, loc_list):
        """ test an image """
        num_cells = 0
        correct_cells = 0
        wd_sz = (self._window_size - 1) / 2
        original_im = self.imread(image, 1)
        segments = self.segment(self.preprocess(original_im), original_im)
        im_width = original_im.shape[1]
        im_height = original_im.shape[0]
        segments = [s for s in segments if s.center[0] -wd_sz >= 0 and s.center[0] + wd_sz +1 < im_height]
        segments = [s for s in segments if s.center[1] -wd_sz >= 0 and s.center[1] + wd_sz +1 < im_width]
        for seg in segments:
            cv2.circle(original_im, seg.center, 3, (255, 0, 0), -1)
            cropped_im = original_im[seg.center[0] - wd_sz: seg.center[0] + wd_sz + 1,
                                     seg.center[1] - wd_sz: seg.center[1] + wd_sz + 1]
            testing_sample = self.extract(cropped_im)

            result = self._classification.predict(testing_sample)
            if result[0] == 1:
                _, value = com.nearest_point(seg.center, loc_list)
                if value <= allidb.tol:
                    correct_cells += 1
                num_cells += 1
        com.debug_im(original_im, True)
        print "The expected number of cells: ", len(loc_list)
        print "correct cells ", correct_cells
        return num_cells

    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
