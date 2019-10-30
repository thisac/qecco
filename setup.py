import codecs
import os
import re

from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_PATH = os.path.join("qecco", "__init__.py")
META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


NAME = "qecco"
DESCRIPTION = find_meta("description")
LICENCE = find_meta("license")
URL = find_meta("url")
VERSION = find_meta("version")
AUTHOR = find_meta("author")
AUTHOR_EMAIL = find_meta("email")
KEYWORDS = ["quantum neural network", "error correction", "bosonic codes"]
LONG = read("README.md")
PACKAGES = find_packages(where=".")
PACKAGE_DIR = {"": "."}
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Physics",
    ]
INSTALL_REQUIRES = [
    "numpy",
    "autograd",
    "scipy",
    "matplotlib",
    "seaborn"
    ]

if __name__ == "__main__":
    setup(
        name=NAME,
        description=DESCRIPTION,
        license=LICENCE,
        url=URL,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        keywords=KEYWORDS,
        long_description=LONG,
        packages=PACKAGES,
        package_dir=PACKAGE_DIR,
        zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        )
