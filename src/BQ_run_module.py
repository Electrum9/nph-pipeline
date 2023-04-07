import nibabel as nib

import csv
import pathlib
import os
import sys
import logging
import cProfile, pstats

import metric
import registration as reg
import postprocess as post
import preprocess as pre
import segmentation as seg

from assets import reference_segmented_scan 

profiler = cProfile.Profile()

def run_module(input_path_dict, output_folder_path):
    profiler.enable()
    scans_path = pathlib.Path(input_path_dict['Input Scans'])
    result_path = output_folder_path / pathlib.Path("results.csv")
    exists = result_path.exists()

    #### BEGIN PIPELINE

    raw_scan = nib.load(scans_path)
    name = scans_path.name.split('-')[-1].split('.')[0]
    print("Loading raw scan: FINISHED")

    (mni_scan, affine) = reg.CT_to_MNI(raw_scan)
    print("CT to MNI for Raw Scan: FINISHED")
    (rest, mask) = pre.skullstrip(mni_scan)
    print("Skullstrip scan: FINISHED")

    ct_rest, inverse_affine = reg.MNI_to_CT(rest, raw_scan, affine)
    ct_mask, _ = reg.MNI_to_CT(mask, raw_scan, reuse=inverse_affine)

    # segmented = seg.inference(ct_rest, ct_mask)
    segmented = nib.load(reference_segmented_scan)
    print("Inference: FINISHED")
    # breakpoint()

    registered_seg = reg.apply_affine(segmented, affine)
    print("Segmented to MNI: FINISHED")
    corrected = post.correct(registered_seg)

    (final_img, inverse_affine) = reg.MNI_to_CT(corrected, raw_scan, reuse=inverse_affine)

    nib.save(mni_scan, "mni_scan.nii.gz")
    nib.save(rest, "rest.nii.gz")
    nib.save(mask, "mask.nii.gz")
    nib.save(segmented, "segmented.nii.gz")
    nib.save(registered_seg, "registered_seg.nii.gz")
    nib.save(corrected, "corrected.nii.gz")
    nib.save(final_img, "final.nii.gz")

    #### END PIPELINE 

    with open(result_path, mode='a+') as result_file:
        # result_writer = csv.writer(result_file, delimiter=',')
        result = metric.compute_metric(final_img, name)

        result_writer = csv.DictWriter(result_file, fieldnames=result.keys())

        if not exists: result_writer.writeheader()

        result_writer.writerow(result) # column names

    output_paths_dict = {'Output Metric': str(result_path)}

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('stats')

    return output_paths_dict

def main():
    input_path_dict = {'Input Scans': sys.argv[1]}
    output_folder_path = pathlib.Path(sys.argv[2])

    output_paths_dict = run_module(input_path_dict, output_folder_path)

    print(output_paths_dict)

if __name__ == "__main__":
    main()
