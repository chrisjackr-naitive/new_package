import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name = "new_package",
      version = "0.1.0",
      author = "Naitive2022",
      description = "Short description...",
      long_description=read('README.md'),
      package_dir = {"": "src"},
      packages=find_packages(where='src',  # '.' by default
                             include=['new_package*']),  # ['*'] by default
                             #exclude=['new_package.tests'],  # empty by default
      install_requires=[],
     )