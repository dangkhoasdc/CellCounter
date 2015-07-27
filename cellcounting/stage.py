"""
File: stage.py
Author: Khoa Le Tan Dang <letan.dangkhoa@gmail.com>
Email: letan.dangkhoa@gmail.com
Github: dangkhoasdc
Description: The Stage Class
"""


class Stage(object):
    """ Stage Class """
    _default_params = dict()

    def __init__(self, name, params=None):
        self.name = name
        if params is None:
            self.params = self._default_params
        elif hasattr(self, "_default_params"):
            self.make_default_params(params)
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

    def set_param(self, param, value):
        """ set a value for the parameter """
        self.params[param] = value
