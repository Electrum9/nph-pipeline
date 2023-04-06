import os
import csv
import numpy as np
import nibabel as nib
import logging

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
    logging.info(f"mm={mm}")

    final_N = final_img.get_fdata()

    final_N_counts = ((final_N == 1) | (final_N == 6)).sum(axis=(0,1))
    logging.info(f"final_N_counts={final_N_counts}")

    max_z = np.argmax(final_N_counts) # slice with highest count
    logging.info(f"max_z={max_z}")

    numSlice = int((35 // final_img.header['pixdim'][3]) // 2)
    logging.info(f"numSlice={numSlice}")

    total_brain_count = (final_N != 1).sum() # everything but 1
    logging.info(f"total_brain_count={total_brain_count}")

    focused_final_N = final_N[...,max_z - numSlice : max_z + numSlice + 1]
    logging.info(f"focused_final_N={focused_final_N}")

    ventricle_count_7 = ((focused_final_N == 1) | (focused_final_N == 6)).sum() # class 1,6
    logging.info(f"ventricle_count_7={ventricle_count_7}")
    logging.info(f"ventricle counts = {((focused_final_N == 1) | (focused_final_N == 6)).sum(axis=(0,1))}")

    csf_count_7 = ventricle_count_7 + (focused_final_N == 3).sum() # classes 1,3,6
    logging.info(f"csf_count_7={csf_count_7}")

    brain_count_7 = (focused_final_N != 0).sum()
    logging.info(f"brain_count_7={brain_count_7}")

    #name = file.name.split('-')[-1].split('.')[0]

    results = {'Name': name,
               'Total Brain Volume': int(total_brain_count*mm), 
               'Center Ventricle Volume': int(ventricle_count_7*mm),
               'Center Brain Volume': int(brain_count_7*mm), 
               'Ventricular Volume Metric': np.round(ventricle_count_7/brain_count_7, 3)}

    return results
