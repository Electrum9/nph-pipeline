import sys
import os
import numpy as np
import pathlib
import fsl.wrappers as fl
import nibabel as nib
from assets import MNI_152_bone, MNI_152

def bone_extracted(ct_img, return_mask=False):
    """ Extract the bone of the CT scan based on the hard thresholding on pixel value.

    Parameters
    ---------- 

    ct_img: nibabel.nifti1.Nifti1Image
        The raw scan.

    return_mask: bool
        Option for returning the mask for regions containing bone pixels, if desired.

    Returns
    ------

    bone_regions: nibabel.nifti1.Nifti1Image
        Subsections of the raw scan, containing bone-like regions.

    tuple[nibabel.nifti1.Nifti1Image, numpy.ndarray]
        Subsections of the raw scan containing bone-like regions, and corresponding binary mask.

    """
    
    ct_img_data = ct_img.get_fdata()

    #print("min = ", np.amin(ct_img_data))
    #print("max = ", np.amax(ct_img_data))

    bone_regions = np.zeros_like(ct_img_data)

    bone_pixel_threshold = 500 # bone pixel threshold

    bone_mask = ct_img_data >= bone_pixel_threshold
    bone_regions[bone_mask] = ct_img_data[bone_mask]


    output = nib.Nifti1Image(bone_regions, ct_img.affine, ct_img.header) # preserve all other info of scan

    if return_mask:
        return output, bone_mask
    
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

def CT_to_MNI(ct_scan, res_path=fl.LOAD, affine_mtx_path=fl.LOAD, bone=None, apply_transformation=True):
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

    bone:
        The bone regions of the scan (skull-only regions). If this is set to
        None, then a (primitive) skull-stripping procedure will be performed.

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

    if bone is None:
        bone = bone_extracted(ct_scan)
    elif pathlike.Path(bone).exists():
        bone = nib.load(bone).get_fdata()

    mtx = fl.flirt(src=bone, ref=MNI_152_bone, omat=affine_mtx_path, bins=256, searchrx=(-180,180), searchry=(-180,180), searchrz=(-180,180), dof=12, interp='trilinear')

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
