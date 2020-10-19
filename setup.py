from __future__ import absolute_import, print_function
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'seggradcam','version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup (
    name='seggradcam',
    version=__version__,
    description='Seg-Grad-CAM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kiraving/SegGradCAM',
    author='Kira Vinogradova',
    author_email='vinograd@mpi-cbg.de',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'csbdeep==0.4.0',
        'tensorflow==1.14.0', #>=1.14.0,<2.0.0',
        'tensorflow-gpu==1.14.0',
        #'keras-gpu>=2.0.0',
        'opencv-python',
    ],
)
