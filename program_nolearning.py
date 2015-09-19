"""
File: program_nolearning.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Program version 2
"""
import argparse
from hed_bilateral import HedBilateralFilter
from segment_hist import SegmentStage
from cellcounting.db import allidb
from cellcounting.fw import NoLearningFramework


def run_program(ftrain, param, param2, viz):
    """ run the program """
    ##########################################
    ## CONFIGURATION
    #################
    num_correct_items = 0
    num_detected_items = 0
    filter_kernel = (param, param)
    sigma_color = param2

    ##########################################
    ## INIT THE FRAMEWORK
    #################
    pre = HedBilateralFilter(filter_kernel, sigma_color)
    seg = SegmentStage(10)
    db = allidb.AllIdb()

    framework = NoLearningFramework(db, pre, seg)

    image_lst, loc_lst = allidb.load_data(ftrain, db.scale_ratio)
    num_true_items = sum([len(loc) for loc in loc_lst])

    for image, loc in zip(image_lst, loc_lst):
        correct, detected= framework.run(image,
                                         loc,
                                         visualize=viz)

        num_correct_items += correct
        num_detected_items += detected

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

    result = run_program(db, 7, 11, viz)
    print result

    if config_output:
        f = open(args["output"], "w")
        f.write(str(result))
        f.close()
