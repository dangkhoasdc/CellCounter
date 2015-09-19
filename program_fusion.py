"""
File: program.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The main program of automatic counting of cells
"""

import sys
import argparse
from hed_bilateral import HedBilateralFilter
from segment_hist import SegmentStage
from cellcounting.db import allidb
from cellcounting.fw import FusionFramework
from cellcounting.features.HedHistHog import HedHistHog


if __name__ == '__main__':
    try:
        ftrain = sys.argv[1]
        ftest = sys.argv[2]
    except:
        ftrain = "train/allidb1_1.txt"
        ftest = "test/allidb1_1.txt"

    total_correct_detected = 0
    total_segments = 0

    operator = "sum"
    filter_kernel = (7, 7)
    sigma_color = 11.0
    preprocessor = HedBilateralFilter(filter_kernel, sigma_color)
    segmentation = SegmentStage(10)
    database = allidb.AllIdb()
    feature_extract = HedHistHog()

    framework = FusionFramework(database,
                                preprocessor,
                                segmentation,
                                feature_extract,
                                operator)

    image_lst, loc_lst, test_images, test_locs = allidb.load_train_test_data(ftrain,
                                                                             ftest,
                                                                             database)

    framework.train(image_lst, loc_lst, False)

    for im, loc in zip(test_images, test_locs):
        total_segments_img, correct_per_img = framework.test(im, loc, True)
        total_correct_detected += correct_per_img
        total_segments += total_segments_img

    print "Accuracy score: ", float(total_correct_detected) / total_segments
