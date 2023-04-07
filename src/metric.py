import os
import csv
import numpy as np
import nibabel as nib

def compute_metric(final_img, name):
    """ Computes various metrics of interest from the given image.

    Specifically, the metrics computed are the Total Brain Volume, Center Ventricle Volume, Center Brain Volume,
    and the Center Ventricle to Brain Volume Ratio (aka the Ventricular Volumetric Metric).

    Parameters
    ----------

    final_img: nibabel.Nifti1Image
        The brain scans that the metrics will be computed for.

    name: str
        Name of the subject the brain scans correspond to.

    Returns
    -------

    dict[str, str | int | float]:
        Dictionary mapping names of each metric to their respective values.
    """


    #final_img = nib.load(file)
    mm = abs(final_img.header['pixdim'][1] * final_img.header['pixdim'][2] * final_img.header['pixdim'][3])

    final_N = final_img.get_fdata()

    final_N_counts = ((final_N == 1) | (final_N == 6)).sum(axis=(0,1))

    max_z = np.argmax(final_N_counts) # slice with highest count

    numSlice = int((35 // final_img.header['pixdim'][3]) // 2)

    total_brain_count = (final_N != 1).sum() # everything but 1

    focused_final_N = final_N[...,max_z - numSlice : max_z + numSlice + 1]

    ventricle_count_7 = ((focused_final_N == 1) | (focused_final_N == 6)).sum() # class 1,6

    csf_count_7 = ventricle_count_7 + (focused_final_N == 3).sum() # classes 1,3,6

    brain_count_7 = (focused_final_N != 0).sum()

    #name = file.name.split('-')[-1].split('.')[0]

    results = {'Name': name,
               'Total Brain Volume': int(total_brain_count*mm), 
               'Center Ventricle Volume': int(ventricle_count_7*mm),
               'Center Brain Volume': int(brain_count_7*mm), 
               'Ventricular Volume Metric': np.round(ventricle_count_7/brain_count_7, 3)}

    return results
