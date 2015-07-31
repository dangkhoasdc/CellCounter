"""
File: program_nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Program version 2
"""
import cv2
import sys
from hed_bilateral import HedBilateralFilter
from segment_hist import SegmentStage
from cellcounting.db import allidb
from cellcounting.fw import nolearning


def run_program(param, param2):
    """ run the program """

    scale = 1 / 6.0
    pre = HedBilateralFilter()
    seg = SegmentStage(3)
    db = allidb.AllIdb()
    framework = nolearning.NoLearningFramework(db, pre, seg)
    filter_kernel = (param, param)
    pre.set_param("bilateral_kernel", filter_kernel)
    pre.set_param("sigma_color", param2)

    try:
        ftrain = sys.argv[1]
    except:
        ftrain = "training.txt"

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
                                                      visualize=True)

        num_correct_items += correct_items
        num_detected_items += detected_items
        cv2.destroyAllWindows()

    R_ir = num_correct_items / float(num_true_items)
    P_ir = num_correct_items / float(num_detected_items)
    perf_ir = 2 * (P_ir * R_ir) / (P_ir + R_ir)
    return perf_ir

if __name__ == '__main__':
    print "Main Program"
    f = open("experiments/auto_canny.csv", "w")
    result = run_program(11, 11)
    print result
    f.write(str(result) + ",")
    f.close()
