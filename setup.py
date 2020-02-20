from setuptools import find_packages
from setuptools import setup

setup(name='globalsalience',
      version='0.1',
      url='https://github.com/alexhernandezgarcia/global-salience',
      install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
          'pyyaml',
          'tqdm',
      ],
      packages=find_packages())
