"""
File: learning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Learning Framework
"""
import cv2
import numpy as np
import cellcounting.common as com
from cellcounting.fw.absframework import AbsFramework

class LearningFramework(AbsFramework):
    def __init__(self,database,
                 preprocess_stage,
                 segmentation_stage,
                 extraction,
                 classifier):
        """init"""
        super(LearningFramework, self).__init__(database,
                                        preprocess_stage,
                                        segmentation_stage)

    def preprocess_segment(self, segment):
        segment.lt[0] -= 4
        segment.lt[1] -= 4
        segment.rb[0] += 3
        segment.rb[1] += 3

        length = max(segment.width, segment.height)

        # update new coordinates
        return segment

    def get_data(self, image_lst, loc_lst, visualize=False):
        """
        prepare data for training phase
        """
        data = []
        labels = []
        result = []

        for image, locs in zip(image_lst, loc_lst):

            demo_img = self.imread(image, 1)
            processed_img = self.preprocess(demo_img)
            segments = self.segment(processed_img)
            segments =[self.preprocess_segment(s) for s in segments]
            locations = list(loc_lst)
            # draw all counted objects in the image
            # visualize true cells
            # check if each segment is close to one true cell

            for seg in segments:
                data.append(seg.get_region(demo_img))
                result.append(seg)
            self.eval_segments(segments, locs)
            labels.extend([1 if s.detected else 0 for s in segments])

            if visualize:
                for loc in locations:
                    cv2.circle(demo_img, loc, 2, (0, 255, 0), 1)
                self.visualize_segments(demo_img, segments, locations)
                com.debug_im(processed_img)
                com.debug_im(demo_img, True)

        return data, labels, result

    def train(self, image_lst, loc_lst, viz=False):
        """
        The training phase
        """
        raise NotImplementedError("This method must be implemented")


    def test(self, image, loc_lst, viz=False):
        """ test an image """
        raise NotImplementedError("This method must be implemented")

