===
dcm
===

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


Running Tests
=============

Test dependencies can be installed with the '[tests]' extra (e.g.  You can then run 
``pytest dcm/`` in this directory. This will also check mypy for errors.
