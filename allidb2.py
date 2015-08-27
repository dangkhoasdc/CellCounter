"""
File: allidb2.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Framework for ALL-IDB2
"""
import sys
import cv2
import numpy as np
from skimage.color import rgb2hed
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
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

# Randomized PCA
n_components = 100
pca = RandomizedPCA(n_components, whiten=True).fit(data_train)

pca_features_train = pca.transform(data_train)
pca_features_test = pca.transform(data_test)

grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
              "C": np.logspace(-5, -3, num=8, base=2),
              "gamma": np.logspace(-15, 3, num=8, base=2)}

clf = GridSearchCV(SVC(class_weight="auto"), grid_param)
clf = clf.fit(pca_features_train, labels_train)

print clf.best_estimator_

y_pred = clf.predict(pca_features_test)

score = accuracy_score(labels_test, y_pred)
print score

