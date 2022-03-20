from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.1.1'

REQUIRED_PACKAGES = []
DEPENDENCY_LINKS = []

setuptools.setup(
    name='adaptive_warmup',
    version=_VERSION,
    description='Adaptive gradual warmup for PyTorch',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://parsed.dev/',
    license='MIT License',
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)
