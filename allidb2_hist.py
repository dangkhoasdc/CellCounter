"""
File: allidb2.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Framework for ALL-IDB2
"""
import sys
import cv2
import csv
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2hed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA, SparsePCA, IncrementalPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from hed_bilateral import HedBilateralFilter
from segment_hist import  SegmentStage
from cellcounting import common as com
from cellcounting.features.hog import HOGFeature
from cellcounting.features.lbp import LBP

# Create samples and its label
preprocessor = HedBilateralFilter()
filter_kernel = (7, 7)
preprocessor.set_param("bilateral_kernel", filter_kernel)
preprocessor.set_param("sigma_color", 9)
segment = SegmentStage(5)
hog = HOGFeature()
lbp = LBP(2, 16, "var")
labels = []

def load_list_files(filename):
    """
    load all data form the file of list data
    """
    files = []
    data = []
    hist_data = []
    labels = []
    with open(filename, "r") as f:
        files = [line.strip() for line in f.readlines()]

    for img in files:
        label = img[-5:-4]
        labels.append(float(label))
        image = cv2.imread(img, flags=1)
        hed = cv2.split(rgb2hed(image))[1]
        hed = img_as_ubyte(hed)
        hist = cv2.calcHist([hed], [0], None, [256], [0, 256]).flatten()
        hist_data.append(hist)
        _, region = preprocessor.run(image)
        region = cv2.resize(region, (256, 256))
        lbp_hist, lbp_img = lbp.compute(region)
        # com.debug_im(lbp_img)
        lbp_img = cv2.resize(lbp_img, (30, 30))
        lbp_img = np.nan_to_num(lbp_img)
        data.append(lbp_img.flatten())
        # data.append(lbp_hist)

    data = np.array(data, dtype=np.float32)
    hist_data = np.array(hist_data, dtype=np.float32)
    return data, labels, hist_data

try:
    train_files = sys.argv[1]
    test_files = sys.argv[2]
except:
    train_files = "allidb2_train.txt"
    test_files = "allidb2_test.txt"

data_train, labels_train, hist_train = load_list_files(train_files)
data_test, labels_test, hist_test = load_list_files(test_files)
n_samples = data_train.shape[0]
################################################
# PREPROCESSING
# mean_img = np.mean(data_train, axis=0)
# data_centered = data_train - mean_img
# data_train -= data_centered.mean(axis=1).reshape(n_samples, -1)
data_train = hist_train
data_test = hist_test
# data_train, data_test, labels_train, labels_test = train_test_split(
    # data, labels, test_size=0.25)
# data_train = np.array(data_train, dtype=np.float64)
# data_test = np.array(data_test, dtype=np.float64)
print "train data: ", data_train.shape
print "test data: ", data_test.shape
# Randomized PCA
# components_range = range(2, data_train.shape[1], 2)
# components_range = range(2, 120, 2)

scores = []
# for n_components in components_range:
    # # pca = SparsePCA(n_components, n_jobs=-1).fit(data_train)
    # pca = RandomizedPCA(n_components, whiten=True).fit(data_train)
    # # pca = IncrementalPCA(n_components, whiten=True).fit(data_train)

    # print "hist train: ", hist_train.shape
    # print "hist test: ", hist_test.shape
    # pca_features_train = pca.transform(data_train)
    # pca_features_test = pca.transform(data_test)
pca_features_train = normalize(data_train, axis=1)
pca_features_test = normalize(data_test, axis=1)
grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
                "C": np.logspace(-5, -3, num=8, base=2),
                "gamma": np.logspace(-15, 3, num=8, base=2)}

clf = GridSearchCV(SVC(class_weight="auto"), grid_param,
                    n_jobs=-1,
                    cv=3)
clf = clf.fit(pca_features_train, labels_train)

# print clf.best_estimator_

y_pred = clf.predict(pca_features_test)

score = accuracy_score(labels_test, y_pred)
scores.append(score)
print max(scores)
##########################################
# Write experiment results to file
##

# f = open("./experiments/allidb2_histcolor_55.csv", "wt")
# try:
    # writer = csv.writer(f)
    # writer.writerow(("n_components", "accuracy"))
    # for component, score in zip(components_range, scores):
        # writer.writerow((component, score))

# finally:
    # f.close()

