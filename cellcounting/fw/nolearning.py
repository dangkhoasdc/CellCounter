"""
File: nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Find contours in an image and count how many cells in
this image without using machine learning technique
"""
from .. import common as com
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
        segments = self.eval_segments(segments, loc_list)
        num_cells = len(segments)
        correct = len(filter(lambda x: x.detected, segments))
        if visualize:
            com.visualize_segments(img, segments, locations_list)
            # draw all counted objects in the image
            print "The number of expected cells: ", expected_nums
            print "The number of cells counting by the program:", num_cells
            print "The number of true counting cells: ", correct
            com.debug_im(processed_img)
            com.debug_im(img)

        return correct, num_cells
