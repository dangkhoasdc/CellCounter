"""
File: preprocess.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: Stage Class Definition
"""
from ... import common as com

class Stage(object):
    """ Stage Class """
    _default_params = dict()

    def __init__(self, name, params=None):
        self.name = name
        if params is None:
            self.params = self._default_params
        elif hasattr(self, "_default_params"):
            self.make_default_params(self.params)
        else:
            raise NotImplementedError("Subclass should define _default_params")

    def __str__(self):
        txt = "\n".join([k+":"+str(v) for k, v in self.params.items()])
        return self.name + "\n" + txt

    def run(self, image):
        """ run preprocessing stage """
        raise NotImplementedError("Subclass should be implement this")

    def make_default_params(self, params):
        """ make a default value for params """
        assert len(self._default_params) >= len(params)
        self.params = params
        for def_key in self._default_params:
            if def_key not in self.params:
                self.params[def_key] = self._default_params[def_key]


class Framework(object):
    """Main Framework"""
    def __init__(self, preprocess_stage,
                 segmentation_stage,
                 extraction,
                 classification):
        """init"""
        self._preprocess = preprocess_stage
        self._segmentation = segmentation_stage
        self._classification = classification
        self._extraction = extraction
        self.preprocess_method = None
        self.extract_method = None

    def preprocess(self, image):
        return self._preprocess.run(image)

    def extract_features(self, image):
        return self._extraction.run(image)

    def segment(self, image):
        return self._segmentation.run(image)

    def train(self, image_lst, loc_lst):
        assert type(image_lst) is list
        assert type(loc_lst) is list

        print "Training Phase"
        # Preprocessing
        images = [self.preprocess(im) for im in image_lst]
        # Segmentation
        segments = [self.segment(im) for im in images]
        # Cropping from segmented image
        assert len(segments) != 0

        for im, cords in zip(segments, loc_lst):
            print "dummy"

    def __str__(self):
        return "\n".join(map(str, [
            self._preprocess,
            self._segmentation,
            self._extraction,
            self._classification
        ]))
