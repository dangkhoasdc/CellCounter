#!/usr/bin/env python

from setuptools import setup

setup(name='CellCounter',
      version='1.0',
      description='The automated cell counter',
      author='Khoa Le Tan Dang',
      author_email='letan.dangkhoa@gmail.com',
      url="http://",
      install_requires = [
      	"numpy",
      	"pymorph >= 0.96",
      	"scipy",
      	"scikit-image",
      	"scikit-learn",
      	"matplotlib"
      ]
     )
