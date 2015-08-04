"""
File: add_loc.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Add the location of cells into xyc files
"""
import sys


if __name__ == '__main__':
    f = open(sys.argv[1], "r")
    for line in f.readlines():
        xyc = open(line.strip() + ".xyc", "a")
        while True:
            loc = str(raw_input(""))
            if loc == "":
                print "Done ", line.strip()
                xyc.close()
                break
            x, y =[int(s) for s in loc.split()]
            xyc.write(str(x*6) + " " + str(y*6) + "\n")
