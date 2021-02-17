
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = 'dev'
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = 'High level DICOM file/network operations'

# Dependencies
setup_requires = ['pytest-runner']
install_requires = ['pydicom >= 1.0',
                    'pynetdicom >= 1.5',
                    'click',
                    'toml',
                    'tree-format',
                    'janus >= 0.5',
                    'fifolock',
                    'typing_extensions',
                    'rich',
                    'dateparser',
                    ]
dependency_links = \
    ['https://github.com/pydicom/pynetdicom3/tarball/master#egg=pynetdicom3']
tests_require = ['pytest', 
                 'pytest-asyncio', 
                 'pytest-mypy', 
                 'mypy', 
                 'psutil',
                 'tox',
                 'tox-pyenv']

# Extra requirements for building documentation
extras_requires = {'doc':  ["sphinx", "numpydoc"],
                   'tests': tests_require,
                  }


NAME                = 'dcm'
AUTHOR              = "Brendan Moloney"
AUTHOR_EMAIL        = "moloney@ohsu.edu"
MAINTAINER          = "Brendan Moloney"
MAINTAINER_EMAIL    = "moloney@ohsu.edu"
DESCRIPTION         = description
LICENSE             = "MIT license"
CLASSIFIERS         = CLASSIFIERS
PLATFORMS           = "OS Independent"
ISRELEASE           = _version_extra == ''
VERSION             = __version__
SETUP_REQUIRES      = setup_requires
INSTALL_REQUIRES    = install_requires
TESTS_REQUIRE       = tests_require
EXTRAS_REQUIRES     = extras_requires
DEPENDENCY_LINKS    = dependency_links
PROVIDES            = ["dcm"]
