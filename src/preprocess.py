import nibabel as nib
import numpy as np
import os
from assets import prob_map
import logging

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

    logging.info(f"nii_image.shape = {nii_image.shape}")
    logging.info(f"nii_image is empty: {not np.any(nii_image)}")
    logging.info(f"nii_mask_image.shape = {nii_mask_image.shape}")

    return nii_image, nii_mask_image

#def skullstrip(scan):
#    """ Original implementation
#    """

#    scan_data = scan.get_fdata().copy()
    
#    # after conversion apply MNI mask
#    scan_data[np.where(mask >= 0.3)] = 0

#    #generate binary mask
#    mask_ret = scan_data.copy()
#    mask_ret[np.where(mask >= 0.3)] = 0
#    mask_ret[np.where(mask_ret <= 0)] = 0
#    mask_ret[np.where(mask_ret > 0)] = 1

#    header = scan.header

#    nii_image = nib.Nifti1Image(scan_data.astype(np.float32), affine=None, header=header)
#    nii_mask_image = nib.Nifti1Image(mask_ret.astype(np.float32), affine=None, header=header)

#    return nii_image, nii_mask_image
