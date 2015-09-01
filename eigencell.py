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
from skimage.color import rgb2hed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, IncrementalPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cellcounting.common as com

try:
    filename = sys.argv[1]
except:
    filename = "allidb2.txt"

files = []

with open(filename, "r") as f:
    files = [line.strip() for line in f.readlines()]

# Create samples and its label
labels = []
data = []

for img in files:
    label = img[-5:-4]
    labels.append(float(label))
    image = cv2.imread(img, flags=1)
    image = cv2.split(rgb2hed(image))[1]
    image = cv2.resize(image, (40, 40))
    data.append(image.flatten())

data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.25)
print "test size: ", len(data_test)
print "train size: ", len(data_train)
data_train = np.array(data_train)
# data_train = np.transpose(data_train)
data_test = np.array(data_test)
# data_test = np.transpose(data_test)
# Randomized PCA
n_components = 400
# pca = SparsePCA(n_components, n_jobs=-1).fit(data_train)
# pca = RandomizedPCA(n_components, whiten=True).fit(data_train)
print data_train.shape
pca = PCA(n_components=400).fit(data_train)
print "components size: ", pca.components_.shape
pca_features_train = pca.transform(data_train)
pca_features_test = pca.transform(data_test)
print pca_features_train.shape

for feature in pca_features_train:
    com.debug_im(feature.reshape(20, 20))

# grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
                # "C": np.logspace(-5, -3, num=8, base=2),
                # "gamma": np.logspace(-15, 3, num=8, base=2)}

# clf = GridSearchCV(SVC(class_weight="auto"), grid_param)
# clf = clf.fit(pca_features_train, labels_train)

# print clf.best_estimator_

# y_pred = clf.predict(pca_features_test)

# score = accuracy_score(labels_test, y_pred)
# scores.append(score)
