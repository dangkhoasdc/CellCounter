"""
File: program_nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Program version 2
"""
import cv2
import sys
import argparse
from hed_bilateral import HedBilateralFilter
from segment_hist import SegmentStage
from cellcounting.db import allidb
from cellcounting.fw import nolearning


def run_program(ftrain, param, param2, viz):
    """ run the program """

    scale = 1 / 6.0
    pre = HedBilateralFilter()
    seg = SegmentStage(10)
    db = allidb.AllIdb()
    framework = nolearning.NoLearningFramework(db, pre, seg)
    filter_kernel = (param, param)
    pre.set_param("bilateral_kernel", filter_kernel)
    pre.set_param("sigma_color", param2)

    file_train = open(ftrain, "r")
    row_lst = [line.strip() for line in file_train.readlines()]
    file_train.close()

    image_lst = [line+".jpg" for line in row_lst]
    loc_lst = [allidb.get_true_locs(line+".xyc", scale) for line in row_lst]
    num_correct_items = 0
    num_detected_items = 0
    num_true_items = sum([len(loc) for loc in loc_lst])

    for image, loc in zip(image_lst, loc_lst):
        print image
        correct_items, detected_items = framework.run(image,
                                                      loc,
                                                      visualize=viz)

        num_correct_items += correct_items
        num_detected_items += detected_items
        cv2.destroyAllWindows()

    R_ir = num_correct_items / float(num_true_items)
    P_ir = num_correct_items / float(num_detected_items)
    perf_ir = 2 * (P_ir * R_ir) / (P_ir + R_ir)
    return perf_ir

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--database", required=True,
                    help="Path to the database file")
    ap.add_argument("-o", "--output", help="Path to the result file")
    ap.add_argument("-v", "--visualize", action="store_true",
                    help="visualize data")

    args = vars(ap.parse_args())
    config_output = args["output"] is not None
    db = args["database"]
    viz = args["visualize"]
    if config_output:
        f = open(args["output"], "w")

    result = run_program(db, 11, 11, viz)
    print result

    if config_output:
        f.write(str(result) + ",")
        f.close()
