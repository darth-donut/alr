# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
try:
    with open(path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except:
    long_description = ""


# get version from __init__.py
def get_version(rel_path):
    for line in open(rel_path).read().splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="alr",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=get_version(path.join("alr", "__init__.py")),
    description="Active learning research library",
    # Fix windows newlines.
    long_description=long_description.replace("\r\n", "\n"),
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/jiahfong/alr",
    # Author details
    author="Jia Hong Fong",
    author_email="jiahfong@gmail.com",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    # What does your project relate to?
    keywords="tools pytorch",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    #
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "torch==1.7.1",
        "tqdm",
        "numpy",
        "torchvision==0.8.2",
        "pytorch-ignite==0.4.1",
        "batchbald-redux==2.0.2",
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        "dev": ["black", "matplotlib"],
        "test": [
            "coverage",
            "codecov",
            "pytest",
            "pytest-benchmark",
            "pytest-cov",
            "pytest-forked",
        ],
    },
    # setup_requires=["pytest-runner"],
)
