===
dcm
===

.. image:: https://codecov.io/gh/moloney/dcm/branch/main/graph/badge.svg
  :target: https://codecov.io/gh/moloney/dcm

This is a Python package and CLI application for performing high-level DICOM
file and network operations. This includes performing queries, performing
'rsync' like data transfers, routing data (including dynamic routing), sorting
data, dumping / diffing data sets, and more.

The goal for the CLI is to provide a human friendly interface for working with
DICOM data and networking. We hide many of the pitfalls and idiosyncrasies of
the DICOM networking standard behind high level abstractions and sensible
defaults. Command line users can tweak these defaults and access more advanced
features through the powerful configuration system.

The goal for the Python API is similar in many ways with a different audience.
Again we want to provide a high-level and user friendly API that hides many of
the lower level details when appropriate. Obviously the Python API is much more
expansive and flexible. The API we provide leverages asyncio for concurrency
and uses type hints extensively (which are checked by mypy in the test suite).

This project would not be possible without the hard work of the
`pydicom <https://pydicom.github.io>`_ and
`pynetdicom <https://pydicom.github.io>`_ contributors.

Requires Python 3.7+.


CLI Quickstart
==============

After installing the python package you will have the command ``dcm`` available.
Running ``dcm --help`` will give an overview of the CLI options.

Before interacting with most DICOM servers (e.g a PACS) you will need to have
your computer registered by the administrator of that system with a specific
AETitle and port.

You can use the ``conf`` command to edit the `TOML <https://toml.io>`_
configuration file and add your AETitle and port in the ``[local_nodes.default]``
section.  You can also add any PACS or other servers at this point in the
``[remote_nodes]`` section.

You can use the ``echo`` command to test connectivity with a DICOM server:

.. code-block:: console

  $ dcm echo mypacs.org:MYPACS:11112
  Success

Any command that requires DICOM network node information can take it directly
as ``hostname:ae_title:port`` as we have done above, or you can reference nodes
in your config file by name. For excample if our config file contains this
section:

.. code-block:: toml

  [remote_nodes.mypacs]
  hostname = "mypacs.org"
  ae_title = "MYPACS"
  port = 11112

We could rewrite the last command as simply:

.. code-block:: console

  $ dcm echo mypacs
  Success

We can query the remote system using the ``query`` command:

.. code-block:: console

  $ dcm query mypacs StudyDate=20201031

  2 patients | 3 studies | 56 series | 17174 instances
  └── ID: Vamp001 | Name: Dracula^Count | 2 studies | 35 series | 8974 instances
  │   ├── Date: 20201031 | Time: 164315.866000 | 21 series | 8200 instances
  │   └── Date: 20201031 | Time: 203205.137000 | 14 series | 774 instances
  └── ID: Vamp002 | Name: Chocula^Count | 1 studies | 21 series | 8200 instances
      └── Date: 20201031 | Time: 102407.269000 | 21 series | 8200 instances

Once we find the data we want to download we can use the ``sync`` command to transfer
it.  We can even just pipe the output from the ``query`` command into the ``sync`` command:

.. code-block:: console

  $ dcm query mypacs StudyDate=20201031 PatientID=Vamp001 | dcm sync path/to/save/data

This is equivalent to running:

.. code-block:: console

  $ dcm sync -s mypacs -q StudyDate=20201031 -q PatientID=Vamp001 path/to/save/data

The data will be saved into the provided directory, sorted into a directory hierarchy
by patient / study / series.

You can actually provide multiple sources and multiple destinations to the ``sync``
command, and both the sources and and destinations can be either directories or
network nodes. By default, whenever possible, all data that already exists on the
destination is skipped (unless ``--force-all`` is used). For local directories we
have no way of reliably knowing what already exists so we can't do this (adding
a Sqlite database option to manage this is future work).


Python Quickstart
=================

The ``net.LocalEntity`` class provides an high-level async API for most common
DICOM networking tasks.

.. code-block:: pycon

  >>> import asyncio
  >>> from dcm.net import DcmNode, LocalEntity
  >>> local = LocalEntity(DcmNode('0.0.0.0', 'LOCALAE', 11112))
  >>> mypacs = DcmNode('mypacs.org', 'MYPACS', 11112)
  >>> asyncio.run(local.echo(mypacs))
  True

We can use it to perform queries, all at once with the ``query`` method or
through an async generator using the ``queries`` method, which will produce
``query.QueryResult`` objects.

.. code-block:: pycon

  >>> from pydicom import Dataset
  >>> query = Dataset()
  >>> query.StudyDate = '20201031'
  >>> qr = asyncio.run(local.query(mypacs, query=query))
  >>> print(qr.to_tree())

  2 patients | 3 studies | 56 series | 17174 instances
  └── ID: Vamp001 | Name: Dracula^Count | 2 studies | 35 series | 8974 instances
  │   ├── Date: 20201031 | Time: 164315.866000 | 21 series | 8200 instances
  │   └── Date: 20201031 | Time: 203205.137000 | 14 series | 774 instances
  └── ID: Vamp002 | Name: Chocula^Count | 1 studies | 21 series | 8200 instances
      └── Date: 20201031 | Time: 102407.269000 | 21 series | 8200 instances

The easiest way to save this data locally is to use the download method which
will save all the DICOM files into a single directory, using the SOPInstanceUID
to name the files.

.. code-block:: pycon

  >>> asyncio.run(local.download(mypacs, qr, 'path/to/save/data'))

For more control you can use the ``retrieve`` method which is an async generator
that produces the incoming data as ``pydicom.Dataset`` objects.

.. code-block:: pycon

  >>> async def print_incoming(local, remote, qr):
  ...     async for ds in local.retrieve(remote, qr):
  ...         print(ds.SOPInstanceUID)

  >>> asyncio.run(print_incoming(local, mypacs, qr))


Contributing Quickstart
=======================

If your system python is too old, or you want to be able to run the tests locally
against multiple python versions it is recommended that you use
`pyenv <https://github.com/pyenv/pyenv>`_ to manage installed python versions.

We use the newer "pyproject.toml" instead of a "setup.py" (plus a bunch of other
files). Using `poetry <https://python-poetry.org/>`_ to manage dependencies and
virtual environments is highly recommended.

All code should be formatted with the `black <https://github.com/psf/black>`_ code
formatter, and this will be done automatically before each commit by
`pre-commit <https://pre-commit.com/>`_ if you run ``poetry run pre-commit install`` once
from inside your local git repo.

All code should be typed and pass the `mypy <http://mypy-lang.org/>`_ type checker
unless there is a good reason not to.


Running Tests Locally
---------------------

The dependencies needed for testing and development are all listed as poetry
"development dependencies". Doing ``poetry run pytest`` is the easiest way to run
all the tests against the current environment. If you have multiple python versions
setup with pyenv you can do ``poetry run tox`` to run the tests against all versions.

While the mypy checker is run by default by pytest, one advantge of mypy is that it
can run many orders of magnitude faster that the test suite while still catching many
errors. You can do ``poetry run mypy`` to just run the mypy checker.

Many tests will be skipped if `dcmtk <https://dicom.offis.de/dcmtk.php.en>`_ is not
installed as we use it to provide a test server. Using (the recently released)
pynetdicom.apps.qrscp as an alternative test server is a high priority.


Continuous integration
----------------------

We use `github actions <https://github.com/features/actions>`_ to automatically run
the test suite on all pushes and pull requests.
