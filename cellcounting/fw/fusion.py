"""
File: fusion.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Fusion of classifiers
"""
import numpy as np
import cv2
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from cellcounting.fw.learning import LearningFramework
import cellcounting.common as com

class FusionFramework(LearningFramework):
    """
    Fusion of classifiers
    """
    def __init__(self, preprocess_stage,
                 segmentation_stage,
                 database,
                 extraction,
                 operator="sum",
                 n_components=20):
        """init"""
        super(FusionFramework, self).__init__(preprocess_stage,
                                              segmentation_stage,
                                              database,
                                              extraction,
                                              None)
        self.op = operator
        grid_param = {"kernel": ("rbf", "poly", "sigmoid"),
                        "C": np.logspace(-5, -3, num=8, base=2),
                        "gamma": np.logspace(-15, 3, num=8, base=2)}
        self._extraction = extraction

        self._pca = RandomizedPCA(n_components=n_components, whiten=True)
        self.classifer = SVC(class_weight="auto", probability=True)
        self.hist_clf = SVC(class_weight="auto", probability=True)

        self.clf_lbp = GridSearchCV(self.classifer,
                                    grid_param,
                                    n_jobs=-1,
                                    cv=3)


        self.clf_hist = GridSearchCV(self.hist_clf,
                                     grid_param,
                                     n_jobs=-1,
                                     cv=3)


    def get_data(self, image_lst, loc_lst, visualize=False):
        """
        prepare data for training phase
        """
        data = []
        labels = []
        result = []
        print "visualization: ", visualize
        for image, locs in zip(image_lst, loc_lst):

            demo_img = self.imread(image, 1)
            processed_img, gray_img = self.preprocess(demo_img)
            segments = self.segment(processed_img, gray_img, demo_img)

            correct = 0
            # draw all counted objects in the image

            if visualize:
                for seg in segments:
                    seg.draw(demo_img, (0, 255, 0), 1)
                for loc in locs:
                    cv2.circle(demo_img, loc, 2, (0, 255, 0), 1)

            # visualize true cells
            # check if each segment is close to one true cell
            for seg in segments:
                data.append(seg.get_region(demo_img))
                result.append(seg)

                if len(locs) == 0:
                    labels.append(-1)
                    continue

                point, dist = com.nearest_point(seg.center, locs)

                if dist <= self._db.tol:
                    locs.remove(point)
                    correct += 1
                    labels.append(1)
                    if visualize:
                        seg.draw(demo_img, (255, 255, 0), 1)
                else:
                    labels.append(-1)

            if visualize:
                com.debug_im(gray_img)
                com.debug_im(processed_img)
                com.debug_im(demo_img, True)

        return data, labels, result

    def train(self, image_lst, loc_lst, viz=False):
        """
        The training phase
        """
        # get the training data

        data, labels, _ = self.get_data(image_lst, loc_lst, viz)
        print "Get data done"
        n_samples = len(data)
        training_data = [self._extraction.compute(im) for im in data]
        hist_data, hog_data = zip(*training_data)

        hist_data = np.array(hist_data, dtype=np.float32)
        hog_data = np.array(hog_data, dtype=np.float32)
        ############################
        # Normalize the histogram feature
        hist_data = normalize(hist_data, axis=1)

        #############################
        # Normalize the hog feature
        mean_img = np.mean(hog_data, axis=0)
        data_centered = hog_data - mean_img
        hog_data -= data_centered.mean(axis=1).reshape(n_samples, -1)

        ##############################
        # RandomizedPCA for the hog feature
        self._pca.fit(hog_data)
        hog_data = self._pca.transform(hog_data)

        ################################
        # Train data using svm
        self.clf_hist = self.clf_hist.fit(hist_data, labels)
        self.clf_lbp = self.clf_lbp.fit(hog_data, labels)

        print "Train Done"



    def test(self, image, loc_lst, viz=False):
        """ test an image """
        demo = self.imread(image)
        correct = 0
        if viz:
            for loc in loc_lst:
                cv2.circle(demo, loc, 2, (0, 0, 255), 3)

        data, labels, segments = self.get_data([image], [loc_lst])
        total_segments = len(data)
        testing_data = [self._extraction.compute(im) for im in data]
        hist_data, hog_data = zip(*testing_data)

        hist_data = np.array(hist_data, dtype=np.float32)
        hog_data = np.array(hog_data, dtype=np.float32)
        ############################
        # Normalize the histogram feature
        hist_data = normalize(hist_data, axis=1)

        ##############################
        # RandomizedPCA for the hog feature

        hog_data = self._pca.transform(hog_data)
        ##############################
        # Fusion of classifiers

        y_proba_lbp = self.clf_lbp.predict_proba(hog_data)
        y_proba_hist = self.clf_hist.predict_proba(hist_data)
        y_proba = None

        if self.op == "sum":
            y_proba = (y_proba_hist + y_proba_lbp)
        elif self.op == "max":
            y_proba = np.maximum(y_proba_hist, y_proba_lbp)
        elif self.op == "min":
            y_proba = np.minimum(y_proba_hist, y_proba_lbp)
        elif self.op == "mul":
            y_proba = np.multiply(y_proba_hist, y_proba_lbp)

        # print "y proba:", y_proba
        result = np.argmax(y_proba, axis=1)

        # score = accuracy_score(labels_test, predicted)
        # visualization
        for predicted, expected, s in zip(result, labels, segments):
            if predicted == expected:
                correct += 1
                if viz:
                    s.draw(demo, (0, 255, 0), 1)
            else:
                if viz:
                    s.draw(demo, (255, 255, 0), 1)
        if viz:
            com.debug_im(demo)
        return total_segments, correct
