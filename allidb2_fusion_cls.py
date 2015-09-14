"""
File: allidb2_fusion_cls.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Fusion of classifiers
"""
import sys
import cv2
import csv
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2hed
from skimage.filters.rank import enhance_contrast
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral
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

###################################################################
# Configuration

image_size = (256, 256)
lbp_size = (40, 40)
kernel = disk(3)



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
        labels.append(int(label))
        image = cv2.imread(img, flags=1)
        hed = cv2.split(rgb2hed(image))[1]
        hed = denoise_bilateral(hed, sigma_range=0.1, sigma_spatial=1.5)
        hed = img_as_ubyte(hed)
        # hed = enhance_contrast(hed, kernel)
        com.debug_im(hed)
        hist = cv2.calcHist([hed], [0], None, [256], [0, 256]).flatten()
        hist_data.append(hist)
        _, region = preprocessor.run(image)
        region = cv2.resize(region, image_size)
        # hog_img, _ = hog.compute(region)
        lbp_img, _ = lbp.compute(region)
        com.debug_im(_)

        data.append(_)

    data = np.array(data, dtype=np.float32)
    hist_data = np.array(hist_data, dtype=np.float32)
    return data, labels, hist_data

try:
    train_files = sys.argv[1]
    test_files = sys.argv[2]
    op = sys.argv[3]
except:
    train_files = "allidb2_train.txt"
    test_files = "allidb2_test.txt"
    op = "min"

data_train, labels_train, hist_train = load_list_files(train_files)
data_test, labels_test, hist_test = load_list_files(test_files)
n_samples = data_train.shape[0]
print "Operator: ", op
################################################
# PREPROCESSING
mean_img = np.mean(data_train, axis=0)
data_centered = data_train - mean_img
data_train -= data_centered.mean(axis=1).reshape(n_samples, -1)

# data_train, data_test, labels_train, labels_test = train_test_split(
    # data, labels, test_size=0.25)
# data_train = np.array(data_train, dtype=np.float64)
# data_test = np.array(data_test, dtype=np.float64)
# Randomized PCA
# components_range = range(2, data_train.shape[1], 2)
components_range = range(2, 100, 2)

scores = []
grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
                "C": np.logspace(-5, -3, num=8, base=2),
                "gamma": np.logspace(-15, 3, num=8, base=2)}

classifer = SVC(class_weight="auto", probability=True)
clf_lbp = GridSearchCV(classifer,
                        grid_param,
                        n_jobs=-1,
                        cv=3)


clf_hist = GridSearchCV(SVC(class_weight="auto", probability=True),
                        grid_param,
                        n_jobs=-1,
                        cv=3)

hist_train = normalize(hist_train, axis=1)
hist_test = normalize(hist_test, axis=1)

for n_components in components_range:
    pca = RandomizedPCA(n_components, whiten=True).fit(data_train)

    pca_features_train = pca.transform(data_train)
    pca_features_test = pca.transform(data_test)

    clf_lbp = clf_lbp.fit(pca_features_train, labels_train)
    clf_hist = clf_hist.fit(hist_train, labels_train)
    # print clf.best_estimator_

    y_proba_lbp= clf_lbp.predict_proba(pca_features_test)
    y_proba_hist = clf_hist.predict_proba(hist_test)
    y_proba = None

    # print "proba_hist:", y_proba_hist
    # print "proba_lbp:", y_proba_lbp

    if op == "sum":
        y_proba = (y_proba_hist + y_proba_lbp)
    elif op == "max":
        y_proba = np.maximum(y_proba_hist, y_proba_lbp)
    elif op == "min":
        y_proba = np.minimum(y_proba_hist, y_proba_lbp)
    elif op =="mul":
        y_proba = np.multiply(y_proba_hist, y_proba_lbp)

    # print "y proba:", y_proba
    y_pred = np.argmax(y_proba, axis=1)

    score = accuracy_score(labels_test, y_pred)
    scores.append(score)
print max(scores)
##########################################
# Write experiment results to file

f = open("./experiments/allidb2_randomizedpca_fusion_add_55.csv", "wt")
try:
    writer = csv.writer(f)
    writer.writerow(("n_components", "accuracy"))
    for component, score in zip(components_range, scores):
        writer.writerow((component, score))

finally:
    f.close()

