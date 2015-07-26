"""
File: check_loc.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: I see that some files in the dataset
have wrong expected location of cells. This script is to check
if each file in a list of files and visualize all cells in image
"""

from cellcounting.db import allidb
import sys

if __name__ == '__main__':
    try:
        filename = sys.argv[1]
    except:
        filename = "training.txt"

    files = []
    with open(filename, "r") as f:
        files = [line.strip() for line in f.readlines()]

    for sample in files:
        print sample
        points = allidb.get_true_locs(sample)
        allidb.visualize_loc(sample + ".jpg", points, True)


