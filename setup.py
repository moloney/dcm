#!/usr/bin/env python

from setuptools import setup, find_packages
import sys, os

# Most of the relevant info is stored in this file
info_file = os.path.join('dcm', 'info.py')
exec(open(info_file).read())


setup(name=NAME,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      classifiers=CLASSIFIERS,
      platforms=PLATFORMS,
      version=VERSION,
      provides=PROVIDES,
      packages=find_packages('.'),
      setup_requires=SETUP_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      extras_require=EXTRAS_REQUIRES,
      dependency_links=DEPENDENCY_LINKS,
      entry_points = {'console_scripts' : ['dcm = dcm.cli:cli']},
     )
