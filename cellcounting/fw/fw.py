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

            # if there are more than 1 segment in this image
            if locs:
                # visualize true cells
                # check if each segment is close to one true cell
                for seg in segments:
                    data.append(seg.get_region(demo_img))
                    result.append(seg)

                    if len(locs) == 0:
                        labels.append(0)
                        continue

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
                    data.append(seg.get_region(demo_img))
                    labels.append(0)
                    result.append(seg)

            if visualize:
                com.debug_im(gray_img)
                com.debug_im(processed_img)
                com.debug_im(demo_img, True)

        return data, labels, result

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


    def test(self, image, loc_lst):
        """ test an image """
        demo = self.imread(image)
        for loc in loc_lst:
            cv2.circle(demo, loc, 2, (0, 0, 255), 3)

        data, l, segments = self.get_data([image], [loc_lst])

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
                s.draw(demo, (0, 255, 0), 1)
            else:
                s.draw(demo, (255, 255, 0), 1)
        com.debug_im(demo)
        return 0


    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
