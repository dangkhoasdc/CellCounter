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
from cellcounting.fw import fw
from cellcounting.features import sift
from cellcounting.classifier import svm


if __name__ == '__main__':
    try:
        ftrain = sys.argv[1]
        ftest = sys.argv[2]
    except:
        ftrain = "train.txt"
        ftest = "test.txt"

    pre = HedBilateralFilter()
    seg = SegmentStage(10)
    db = allidb.AllIdb()
    sift_feature = sift.SIFTFeature()
    svm_clf = svm.SVM()
    framework = fw.Framework(pre, seg, db, sift_feature, svm_clf)
    filter_kernel = (7, 11)
    pre.set_param("bilateral_kernel", filter_kernel)
    pre.set_param("sigma_color", 9)

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

    framework.train(image_lst, loc_lst, "dict")
    for im, loc in zip(test_image_lst, test_loc_lst):
        framework.test(im, loc)

