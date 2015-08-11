"""
File: preprocess.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Stage Class Definition
"""
from .. import common as com
import cv2
from .absframework import AbsFramework


class Framework(AbsFramework):
    """Main Framework"""
    def __init__(self, database,
                 preprocessing,
                 segmentation,
                 extraction,
                 learning_feature,
                 pooling,
                 classifier):
        """init"""
        super(Framework, self).__init__(database,
                                        preprocessing,
                                        segmentation)
        self._extraction = extraction
        self._learning_feature = learning_feature
        self._pooling = pooling
        self._classifier = classifier

    def extract(self, image):
        return self._extraction.run(image)

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

                    if len(locs) == 0:
                        break

                    point, value = com.nearest_point(seg.center, locs)

                    if value <= self._db.tol:
                        locs.remove(point)
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
        """ train data """
        pass

    def test(self, image, loc_list):
        """ test an image """
        pass

    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
