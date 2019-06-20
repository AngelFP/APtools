from setuptools import setup, find_packages

import aptools

# Read long description
with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines()]


# Main setup command
setup(name='APtools',
      version=aptools.__version__,
      author='Angel Ferran Pousa',
      author_email="angel.ferran.pousa@desy.de",
      description='A collection of tools for accelerator physics',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/AngelFP/VisualPIC',
      license='GPLv3',
      packages=find_packages('.'),
      install_requires=read_requirements(),
      platforms='any',
      classifiers=(
          "Development Status :: 1 - Planning",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Physics",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"),
      )
