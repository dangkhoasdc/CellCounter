"""
File: nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Find contours in an image and count how many cells in
this image without using machine learning technique
"""
from .. import common as com
import cv2
from .absframework import AbsFramework


class NoLearningFramework(AbsFramework):
    """NoLearning Framework class """
    def __init__(self, database, preprocessing, segmentation):
        super(NoLearningFramework, self).__init__(database,
                                           preprocessing,
                                           segmentation)


    def run(self, image, locations_list, visualize=False):

        """ run this framework """
        ##########################
        ## Configuration
        ####################
        loc_list = list(locations_list)
        expected_nums = len(loc_list)

        # FRAMEWORK
        img = self.imread(image, 1)

        processed_img = self.preprocess(img)
        segments = self.segment(processed_img)
        # if there are more than 1 segment in this image
        # visualize true cells
        # check if each segment is close to one true cell
        for seg in segments:
            if not loc_list:
                break
            point, value = com.nearest_point(seg.center, loc_list)
            if value <= self._db.tol:
                loc_list.remove(point)
                seg.detected = True

        num_cells = len(segments)
        correct = len(filter(lambda x: x.detected == True, segments))
        if visualize:
            # draw all counted objects in the image
            for loc in locations_list:
                cv2.circle(img, loc, 2, (0, 255, 0), 1)
            for seg in segments:
                if seg.detected:
                    seg.draw(img, (255, 255, 0), 1)
                else:
                    seg.draw(img, (0, 255, 0), 1)

            print "The number of expected cells: ", expected_nums
            print "The number of cells counting by the program:", num_cells
            print "The number of true counting cells: ", correct
            com.debug_im(processed_img)
            com.debug_im(img)

        return correct, num_cells
