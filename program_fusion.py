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
from cellcounting.fw.fusion import FusionFramework
from cellcounting.features import sift
from cellcounting.classifier import svm
from cellcounting.features.HedHistHog import HedHistHog


if __name__ == '__main__':
    try:
        ftrain = sys.argv[1]
        ftest = sys.argv[2]
    except:
        ftrain = "train.txt"
        ftest = "test.txt"

    op = "sum"
    filter_kernel = (7, 7)
    sigma_color = 11
    pre = HedBilateralFilter()
    seg = SegmentStage(10)
    db = allidb.AllIdb()
    feature_extract = HedHistHog()
    pre.set_param("bilateral_kernel", filter_kernel)
    pre.set_param("sigma_color", sigma_color)

    framework = FusionFramework(pre, seg, db, feature_extract, op)
    file_train = open(ftrain, "r")
    row_lst = [line.strip() for line in file_train.readlines()]
    file_train.close()

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [allidb.get_true_locs(line+".xyc", db.scale_ratio) for line in row_lst]

    file_test = open(ftest, "r")
    row_lst = [line.strip() for line in file_test.readlines()]
    file_test.close()

    test_image_lst = [line+".jpg" for line in row_lst]
    test_loc_lst = [allidb.get_true_locs(line+".xyc", db.scale_ratio) for line in row_lst]

    framework.train(image_lst, loc_lst, False)
    total_correct_detected = 0
    total_segments = 0
    for im, loc in zip(test_image_lst, test_loc_lst):
        total_segments_img, correct_per_img = framework.test(im, loc, True)
        total_correct_detected += correct_per_img
        total_segments += total_segments_img

    print "Accuracy score: ", float(total_correct_detected) / total_segments
