"""
File: program.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The main program of automatic counting of cells
"""
import cv2
import numpy as np
import sys
from cellcounting.db import allidb
from cellcounting.preprocessing import morph
from cellcounting.fw import fw
from cellcounting import common as com

if __name__ == '__main__':
    print "Main Program "
