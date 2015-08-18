"""
File: watershed.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The watershed algorithm
"""
import numpy as np
import cv2
from cellcounting.segmentation.contour import Contour
import cellcounting.common as com
from skimage.morphology import watershed as ws
from skimage.morphology import dilation, erosion, disk
from skimage.util import img_as_bool, img_as_ubyte
from skimage.feature import peak_local_max
from skimage.exposure import rescale_intensity
from skimage.filters.rank import maximum
from skimage.restoration import denoise_bilateral
from cellcounting.preprocessing import morph
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)


def watershed(image):
    """ the watershed algorithm """
    if len(image.shape) != 2:
        raise TypeError("The input image must be gray-scale ")

    h, w = image.shape
    image = cv2.equalizeHist(image)
    image = denoise_bilateral(image, sigma_range=0.1, sigma_spatial=10)
    image = rescale_intensity(image)
    image = img_as_ubyte(image)
    image = rescale_intensity(image)
    # com.debug_im(image)

    _, thres = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)

    distance = ndi.distance_transform_edt(thres)
    local_maxi = peak_local_max(distance, indices=False,
                                labels=thres,
                                min_distance=5)

    # com.debug_im(thres)
    # implt = plt.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
    # plt.show()

    markers = ndi.label(local_maxi, np.ones((3, 3)))[0]
    labels = ws(-distance, markers, mask=thres)
    labels = np.uint8(labels)
    # result = np.round(255.0 / np.amax(labels) * labels).astype(np.uint8)
    # com.debug_im(result)

    segments = []
    for idx in range(1, np.amax(labels) + 1):

        indices = np.where(labels == idx)
        left = np.amin(indices[1])
        right = np.amax(indices[1])
        top = np.amin(indices[0])
        down = np.amax(indices[0])

        # region = labels[top:down, left:right]
        # m = (region > 0) & (region != idx)
        # region[m] = 0
        # region[region >= 1] = 1
        region = image[top:down, left:right]
        cont = Contour(mask=region)
        cont.lt = [left, top]
        cont.rb = [right, down]
        segments.append(cont)

    return segments
