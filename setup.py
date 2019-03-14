# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

with open("README.md") as f:
    readme = f.read()

about = {}
with open(os.path.join('exp_mixture_model', '__version__.py'), 'r') as f:
    exec(f.read(), about)

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    install_requires=requirements,
    url=about['__url__'],
    license=about['__license__'],
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    keywords='exponential mixture model, model selection, normalized maximum likelihood',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)

