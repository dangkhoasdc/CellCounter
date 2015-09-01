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
from skimage import img_as_float
from skimage.color import rgb2hed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA, SparsePCA, IncrementalPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from hed_bilateral import HedBilateralFilter
from segment_hist import  SegmentStage
from cellcounting import common as com
from cellcounting.features.hog import HOGFeature
from cellcounting.features.lbp import LBP

try:
    filename = sys.argv[1]
except:
    filename = "allidb2.txt"

files = []

with open(filename, "r") as f:
    files = [line.strip() for line in f.readlines()]

# Create samples and its label
preprocessor = HedBilateralFilter()
filter_kernel = (7, 7)
preprocessor.set_param("bilateral_kernel", filter_kernel)
preprocessor.set_param("sigma_color", 9)
segment = SegmentStage(5)
hog = HOGFeature()
lbp = LBP()
labels = []
data = []
n_samples = len(files)
for img in files:
    label = img[-5:-4]
    labels.append(float(label))
    image = cv2.imread(img, flags=1)
    region = rgb2hed(image)[1]
    max_cont = cv2.resize(region, (40, 40)).flatten()
    # max_cont, _ = lbp.compute(max_cont)
    data.append(max_cont)

data = np.array(data, dtype=np.float32)
################################################
# PREPROCESSING
mean_img = np.mean(data, axis=0)
data_centered = data - mean_img
data -= data_centered.mean(axis=1).reshape(n_samples, -1)

data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.25)
print "test size: ", len(data_test)
print "train size: ", len(data_train)

# Randomized PCA
components_range = range(2, len(data_test), 1)

scores = []
for n_components in components_range:
    # pca = SparsePCA(n_components, n_jobs=-1).fit(data_train)
    pca = RandomizedPCA(n_components, whiten=True).fit(data_train)
    # pca = IncrementalPCA(n_components, whiten=True).fit(data_train)

    pca_features_train = pca.transform(data_train)
    pca_features_test = pca.transform(data_test)

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

print "best accuracy: ", np.max(scores)
##########################################
# Write experiment results to file
##

f = open("./experiments/allidb2_randomizedpca_2.csv", "wt")
try:
    writer = csv.writer(f)
    writer.writerow(("n_components", "accuracy"))
    for component, score in zip(components_range, scores):
        writer.writerow((component, score))

finally:
    f.close()

