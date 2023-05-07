import nibabel as nib
import numpy as np
import os
from assets import prob_map

mask = nib.load(prob_map['background']).get_fdata()


def skullstrip(scan):
    """ Performs skull stripping on provided scan (in MNI space), using a probability map for the background.

    Parameters
    ----------

    scan: nibabel.nift1.Nifti1Image
        Raw scan that is to be skull stripped. Expected to be in MNI space.

    Returns
    -------

    tuple[nibabel.nifti1.Nifti1Image, nibabel.nifti1.Nifti1Image]:
        Pairing of skull stripped scan, as well as binary mask indicating
        regions of the raw scan that are positive-valued (in intensity)
        and are not of the background.
    """

    scan_data = scan.get_fdata().copy()

    # Apply MNI mask
    scan_data[mask >= 0.3] = 0 

    mask_ret = scan_data.copy()

    # Generate binary mask

    mask_ret[mask_ret <= 0] = 0
    mask_ret[mask_ret > 0] = 1

    header = scan.header

    #affine = np.eye(4)
    nii_image = nib.Nifti1Image(scan_data.astype(np.float32), affine=None, header=header)
    nii_mask_image = nib.Nifti1Image(mask_ret.astype(np.float32), affine=None, header=header)

    return nii_image, nii_mask_image
