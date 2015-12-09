__author__ = 'Heikki Arponen'

# Set USE_CYTHON to True to build extensions using Cython.
# Set it to False to use the C file
USE_CYTHON = True

import sys

from distutils.core import setup
from distutils.extension import Extension
import numpy
#import models.deeprnn

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        print('Cython not found. Using C files.')
        USE_CYTHON = False

base_dir = 'python2'

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("ctools/nonpytools", ["ctools/nonpytools" + ext],
                        include_dirs=[numpy.get_include()])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='DerpRNN',
    #version=models.deeprnn.__version__,
    description='Deep RNN models.',
    author='Heikki Arponen',
    author_email='heikki@quantmechanics.com',
    url='',
    py_modules=['models.deeprnn'],
    ext_modules=extensions,

    #long_description=open('README.txt').read(),

    license="GPLv3",
    # install_requires=['numpy>=1.10.1',
    #                   'theano>=0.7.0',
    #                   'python-midi'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords=['Recurrent Neural Network', 'RNN', 'GRU', 'Restricted Boltzmann Machine',
              'RBM'],
)