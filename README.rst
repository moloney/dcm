===
dcm
===

This is a python package and CLI application for performing high-level DICOM
file and network operations. This includes performing queries and perforiming
'rsync' like data transfers.

Requires Python 3.7+.


CLI Quickstart
==============

After installing the python package you will have the command `dcm` available.
Running `dcm --help` will give an overview of the CLI options.

You can use the `echo` command to test connectivity with a DICOM server:

  $ dcm echo mypacs.com:MYPACS:11112
  Success

The server info can be provided directly as `hostname:ae_title:port` as we have 
done above, or you add the server info to the config file and reference the name
used (see `dcm conf --help`).


