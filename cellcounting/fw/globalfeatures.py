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
from cellcounting.fw.absframework import AbsFramework


class Framework(AbsFramework):
    """Main Framework"""
    def __init__(self, preprocess_stage,
                 segmentation_stage,
                 database,
                 extraction,
                 classifier):
        """init"""
        super(Framework, self).__init__(database,
                                        preprocess_stage,
                                        segmentation_stage)
        self._classifier = classifier
        self._extraction = extraction

    def get_data(self, image_lst, loc_lst, visualize=False):
        """
        prepare data for training phase
        """
        data = []
        labels = []
        result = []
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

            # visualize true cells
            # check if each segment is close to one true cell
            for seg in segments:
                data.append(seg.get_region(gray_img))
                result.append(seg)

                if len(locs) == 0:
                    labels.append(-1)
                    continue

                point, dist = com.nearest_point(seg.center, locs)

                if dist <= self._db.tol:
                    locs.remove(point)
                    correct += 1
                    labels.append(1)
                    if visualize:
                        seg.draw(demo_img, (255, 255, 0), 1)
                else:
                    labels.append(-1)

            if visualize:
                com.debug_im(gray_img)
                com.debug_im(processed_img)
                com.debug_im(demo_img, True)

        return data, labels, result

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
        print "The number of false samples:", len(labels[labels == -1])

        # svm
        print training_data.shape
        self._classifier.auto_train(training_data, labels)
        print "train done"
        return self._classifier


    def test(self, image, loc_lst, viz=False):
        """ test an image """
        demo = self.imread(image)
        correct = 0
        if viz:
            for loc in loc_lst:
                cv2.circle(demo, loc, 2, (0, 0, 255), 3)

        data, labels, segments = self.get_data([image], [loc_lst])
        total_segments = len(data)
        testing_data = [self._extraction.compute(im)[0] for im in data]

        testing_data = np.vstack(testing_data)

        result = self._classifier.predict(testing_data)

        for predicted, expected, s in zip(result, labels, segments):
            if predicted == expected:
                correct += 1
                if viz:
                    s.draw(demo, (0, 255, 0), 1)
            else:
                if viz:
                    s.draw(demo, (255, 255, 0), 1)
        if viz:
            com.debug_im(demo)
        return total_segments, correct

    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
