This is my project during Mitacs internship at the university of Saskatchewan. 

# Introduction

The program automatically detects cells in the [ALL-IDB dataset](http://crema.di.unimi.it/~fscotti/all/).

# Proposed Method
[Updating]

# Requirements

1. numpy >= 1.9 
2. opencv >= 2.4.10
3. skimage >= 0.12
4. matplotlib >= 1.4.3
5. pymorph >= 0.96

# Usage

Run `python program_nolearing.py -h` for more details about using the program. 

## Examples

`python program_nolearing.py -d training.txt -v`: count all images which are located in file `training.txt` and visualize the result.

`python program_nolearing.py -d training.txt -o result.txt`: count all images which are located in file `training.txt` and write the result to file `result.txt` 

To load data from `training.txt` or `problem.txt`, you must download the dataset and copy all images and xyc files into folder *data*. Because of the copyright of the dataset, I can not make a copy version in this repository.

