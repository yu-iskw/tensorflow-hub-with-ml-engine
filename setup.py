from setuptools import find_packages
from setuptools import setup

setup(
    name="trainer",
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "tensorflow-hub",
        "h5py",
    ],
)
