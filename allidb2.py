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
from sklearn.decomposition import RandomizedPCA, SparsePCA, IncrementalPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

# Randomized PCA
components_range = range(2, 300, 6)

scores = []
for n_components in components_range:
    # pca = SparsePCA(n_components, n_jobs=-1).fit(data_train)
    # pca = RandomizedPCA(n_components, whiten=True).fit(data_train)
    pca = IncrementalPCA(n_components, whiten=True).fit(data_train)

    pca_features_train = pca.transform(data_train)
    pca_features_test = pca.transform(data_test)

    grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
                  "C": np.logspace(-5, -3, num=8, base=2),
                  "gamma": np.logspace(-15, 3, num=8, base=2)}

    clf = GridSearchCV(SVC(class_weight="auto"), grid_param)
    clf = clf.fit(pca_features_train, labels_train)

    # print clf.best_estimator_

    y_pred = clf.predict(pca_features_test)

    score = accuracy_score(labels_test, y_pred)
    scores.append(score)

##########################################
# Write experiment results to file
##

f = open("./experiments/allidb2_increpca_orig_.csv", "wt")
try:
    writer = csv.writer(f)
    writer.writerow(("n_components", "accuracy"))
    for component, score in zip(components_range, scores):
        writer.writerow((component, score))

finally:
    f.close()

