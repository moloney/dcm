'''Various utility functions'''
from contextlib import asynccontextmanager

from pydicom.uid import generate_uid


class DicomDataError(Exception):
    '''Base class for exceptions from erroneous dicom data'''


class DuplicateDataError(DicomDataError):
    '''A duplicate dataset was found'''


def make_uid_update_cb(uid_prefix='2.25', add_uid_entropy=None):
    if add_uid_entropy is None:
        add_uid_entropy = []
    if uid_prefix[-1] != '.':
        uid_prefix += '.'
    def update_uids_cb(ds, elem):
        '''Callback for updating UID values except `SOPClassUID`'''
        if elem.VR == 'UI' and elem.keyword != 'SOPClassUID':
            if elem.VM > 1:
                elem.value = [generate_uid(uid_prefix,
                                           [x] + add_uid_entropy)
                              for x in elem.value]
            else:
                elem.value = generate_uid(uid_prefix,
                                          [elem.value] + add_uid_entropy)
    return update_uids_cb


# TODO: Update this to use new Filter class, maybe move to the filter module
def make_edit_filter(edit_dict, update_uids=True, uid_prefix='2.25',
                     add_uid_entropy=None):
    '''Make a filter function that edits some DICOM attributes

    Parameters
    ----------
    edit_dict : dict
        Maps keywords to new values (or None to delete the element)

    update_uids : bool
        Set to False to avoid automatically updating UID values

    add_uid_entropy : list or None
        One or more strings used as "entropy" when remapping UIDs
    '''
    update_uids_cb = make_uid_update_cb(add_uid_entropy)
    def edit_filter(ds):
        # TODO: Handle nested attributes (VR of SQ)
        for attr_name, val in edit_dict.items():
            if val is None and hasattr(ds, attr_name):
                delattr(ds, attr_name)
            elif hasattr(ds, attr_name):
                setattr(ds, attr_name, val)
        if update_uids:
            ds.walk(update_uids_cb)
            if hasattr(ds, 'file_meta'):
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        return ds
    return edit_filter


@asynccontextmanager
async def aclosing(thing):
    try:
        yield thing
    finally:
        await thing.aclose()
