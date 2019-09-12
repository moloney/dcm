'''File system based repository for DICOM data

Allows for simple tracking of DICOM meta data for groups of files, 
making it unnecessary to parse every single file
'''
# TODO: Need way to capture DICOM meta data in directory structure, 
#       in such a way that we can get it back at a later date without
#       parsing every single file again.
#
# Could use hidden files (.dcm_meta) at one or more levels of the dir 
# heirarchy. 
#
#  Treat this as a promise: all dicom files below this dir have the 
#  the following DICOM attributes.
#


class DcmMetaDir(object):
    '''Work with dirs that are augmented with meta data files'''
    def __init__(self, root_dir):
        pass
