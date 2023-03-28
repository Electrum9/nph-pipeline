import nibabel as nib

import csv
import pathlib
import os
import sys
import logging

import metric
import registration as reg
import postprocess as post
import preprocess as pre
import segmentation as seg

def run_module(input_path_dict, output_folder_path):
    scans_path = pathlib.Path(input_path_dict['Input Scans'])
    result_path = output_folder_path / pathlib.Path("results.csv")
    exists = result_path.exists()

    #### BEGIN PIPELINE

    raw_scan = nib.load(scans_path)

    (mni_scan, affine) = reg.CT_to_MNI(raw_scan)
    (rest, mask) = pre.skullstrip(mni_scan)
    segmented = seg.inference(rest, mask)

    registered_seg = reg.apply_affine(segmented, affine)
    corrected = post.correct(registered_seg)

    (final_img, inverse_affine) = reg.MNI_to_CT(corrected, raw_scan, affine)

    #### END PIPELINE 

    with open(result_path, mode='a+') as result_file:
        # result_writer = csv.writer(result_file, delimiter=',')
        result = metric.compute_metric(final_img)

        result_writer = csv.DictWriter(result_file, fieldnames=result.keys())

        if not exists: result_writer.writeheader()

        result_writer.writerow(result) # column names

    output_paths_dict = {'Output Metric': str(result_path)}

    return output_paths_dict

def main():
    input_path_dict = {'Input Scans': sys.argv[1]}
    output_folder_path = pathlib.Path(sys.argv[2])

    output_paths_dict = run_module(input_path_dict, output_folder_path)

    print(output_paths_dict)

if __name__ == "__main__":
    main()
