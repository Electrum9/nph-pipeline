import sys
import os
import numpy as np
import pathlib
import fsl.wrappers as fl
import nibabel as nib

from assets import MNI_152_bone, MNI_152

def basic_skullstrip(ct_img):
    """ Eliminate the bone of the CT scan based on hard thresholding of pixel value.

    Parameters
    ---------- 

    ct_img: nibabel.nifti1.Nifti1Image
        The raw scan.

    Returns
    ------

    output: nibabel.nifti1.Nifti1Image
        Subsections of the raw scan, excluding bone-like regions.

    """
    
    ct_img_data = ct_img.get_fdata()

    #print("min = ", np.amin(ct_img_data))
    #print("max = ", np.amax(ct_img_data))

    brain_regions = np.zeros_like(ct_img_data)

    bone_pixel_threshold = 500 # bone pixel threshold

    brain_mask = ct_img_data <= bone_pixel_threshold # brain only regions (no bone)
    brain_regions[brain_mask] = ct_img_data[brain_mask]

    output = nib.Nifti1Image(brain_regions, ct_img.affine, ct_img.header) # preserve all other info of scan

    return output

def MNI_to_CT(MNI_scan, ct_scan, affine_mtx=None, res_path=fl.LOAD, inv_path=fl.LOAD, reuse=None):
    """Brings scans in MNI space to subject space.

    This requires the segmented scan's raw counterpart scan, as well
    as the affine matrix mapping from subject space to MNI -- 
    this is so it can generate the inverse affine transformation,
    and produce the final result in subject space.

    Can work either with the paths to the scans, or Nifti1Image objects
    in memory.

    Parameters
    ----------

    MNI_scan: nibabel.nifti1.Nifti1Image | pathlib.Path
        The segmented scan in MNI space.

    ct_scan: nibabel.nifti1.Nifti1Image | pathlib.Path
        The segmented scan's raw scan counterpart.

    affine_mtx: numpy.ndarray | pathlib.Path
        The affine matrix mapping from MNI to subject space.

    inv_path: pathlib.Path | object
        Path where the inverse transformation should be saved.

    res_path: pathlib.Path | object
        Path where the resultant transformed scan should be saved.

    reuse: Optional[numpy.ndarray | pathlib.Path]
        Optionally specify an array to be used for the inverse transformation.
        Use this only if the inverse affine transformation was already computed.

    Returns
    -------

    tuple[nibabel.nifti1.Nifti1Image | pathlib.Path, numpy.ndarray | pathlib.Path]:
        Pairing of segmented scan (in subject space) and inverse affine
        transformation.

    """

    if reuse is None:
        assert affine_mtx is not None

        inv_mtx = fl.invxfm(affine_mtx, omat=inv_path) 

        if inv_path != fl.LOAD:
            inv_mtx = inv_path
        else:
            inv_mtx = inv_mtx['omat']
    else:
        inv_mtx = reuse

    reference = fl.applyxfm(MNI_152, ct_scan, inv_mtx, out=fl.LOAD)['out']
    res = fl.applyxfm(MNI_scan, reference, inv_mtx, out=res_path, interp='nearestneighbour') 

    if res_path != fl.LOAD:
        res = res_path
    else:
        res = res['out']

    return res, inv_mtx

def CT_to_MNI(ct_scan, res_path=fl.LOAD, affine_mtx_path=fl.LOAD, brain_regions=None, apply_transformation=True):
    """ Finds transformation from CT to MNI space.

    Parameters
    ----------

    ct_scan: 
        Raw CT scan.

    affine_mtx_path:
        Path where the affine matrix is to be stored. By default, this is
        set to a special value which indicates it should be stored in memory,
        as an object.

    res_path:
        Path where the result is to be stored. By default, this is
        set to a special value which indicates it should be stored in memory,
        as an object.

    brain_regions:
        The brain regions of the scan (no skull). If this is set to
        None, then a primitive skull-stripping procedure will be performed.

    apply_transformation:
        Flag for indicating whether derived affine transformation should be
        applied to given scan.

    Returns
    ----------

    tuple[pathlib.Path | nibabel.nifti1.Nifti1Image, numpy.ndarray]:
        Registered scan, using FSL flirt; numpy array containing matrix used for transformation.

    numpy.ndarray:
        Numpy array containing matrix used for transformation.
        
    """

    if brain_regions is None:
        brain_regions = basic_skullstrip(ct_scan) # improves registration
        # brain_regions = ct_scan # NOTE: Temporarily turned off primitive skull stripping
    else:
        assert isinstance(brain_regions, nib.Nifti1Image)

    mtx = fl.flirt(src=brain_regions, ref=MNI_152, omat=affine_mtx_path, bins=256, searchrx=(-180,180), searchry=(-180,180), searchrz=(-180,180), dof=12, interp='trilinear')

    if affine_mtx_path != fl.LOAD:
        mtx = affine_mtx_path
    else:
        mtx = mtx['omat']

    if not apply_transformation: return mtx

    ct_scan = fl.applyxfm(ct_scan, MNI_152, mtx, res_path, interp='nearestneighbour')

    if res_path != fl.LOAD:
        ct_scan = res_path
    else:
        ct_scan = ct_scan['out']

    return ct_scan, mtx

def apply_affine(scan, affine, res_path=fl.LOAD):

    res = fl.applyxfm(scan, MNI_152, affine, res_path, interp='nearestneighbour')

    if res_path != fl.LOAD:
        res = res_path
    else:
        res = res['out']

    return res


if __name__ == "__main__":
    '''
    Usage:

    ./utilities.py mni ct { MNI_SCAN_PATH } { CT_SCAN_PATH } { AFFINE_MTX_PATH } { RESULT_PATH } [ INVERSE_MTX_PATH ]
    ./utilities.py ct mni { SEGMENTED_SCAN_PATH } { RAW_SCAN_PATH } { RESULT_PATH } [ AFFINE_MTX_PATH ] [ BONE_PATH ]

    Any arguments encased in square brackets [] are optional (they will simply be constructed in memory, and will not be saved as a file).

    '''
    valid_spaces = {'mni', 'ct'}

    # Below dispatches the appropriate implementation of the desired transformation
    trans_impl = {('mni', 'ct'): MNI_to_CT,
                  ('ct', 'mni'): CT_to_MNI,
                 }

    print(sys.argv)
    transform_mapping = tuple(str.lower(x) for x in sys.argv[1:3])

    src_space, tgt_space = transform_mapping # source space -> target space

    assert src_space in valid_spaces
    assert tgt_space in valid_spaces

    params = sys.argv[3:]
    transformation = trans_impl[transform_mapping]

    transformation(*params)
