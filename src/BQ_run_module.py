import nibabel as nib

import csv
import pathlib
import os
import sys

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
    name = scans_path.name.split('-')[-1].split('.')[0]
    print("Loading raw scan: FINISHED")

    (mni_scan, affine) = reg.CT_to_MNI(raw_scan)
    print("CT to MNI for Raw Scan: FINISHED")
    (rest, mask) = pre.skullstrip(mni_scan)
    print("Skullstrip scan: FINISHED")

    ct_rest, inverse_affine = reg.MNI_to_CT(rest, raw_scan, affine)
    ct_mask, _ = reg.MNI_to_CT(mask, raw_scan, reuse=inverse_affine)

    segmented = seg.inference(ct_rest, ct_mask, batch_size=5000)
    print("Inference: FINISHED")
    # breakpoint()

    registered_seg = reg.apply_affine(segmented, affine)
    print("Segmented to MNI: FINISHED")
    corrected = post.correct(registered_seg)

    (final_img, inverse_affine) = reg.MNI_to_CT(corrected, raw_scan, reuse=inverse_affine)

    nib.save(mni_scan, name + "_mni_scan.nii.gz")
    nib.save(rest, name + "_rest.nii.gz")
    nib.save(mask, name + "_mask.nii.gz")
    nib.save(segmented, name + "_segmented.nii.gz")
    nib.save(registered_seg, name + "_registered_seg.nii.gz")
    nib.save(corrected, name + "_corrected.nii.gz")

    final_name = name + "_final.nii.gz"
    nib.save(final_img, final_name)

    #### END PIPELINE 

    # Compute metric (commented out for now, so extra corrections can be applied to segmentation)

    # with open(result_path, mode='a+') as result_file:
    #     # result_writer = csv.writer(result_file, delimiter=',')
    #     result = metric.compute_metric(final_img, name)

    #     result_writer = csv.DictWriter(result_file, fieldnames=result.keys())

    #     if not exists: result_writer.writeheader()

    #     result_writer.writerow(result) # column names

    # output_paths_dict = {'Output Metric': str(result_path)}


    return {'Output Scan': final_name}

def main():
    input_path_dict = {'Input Scans': sys.argv[1]}
    output_folder_path = pathlib.Path(sys.argv[2])

    output_paths_dict = run_module(input_path_dict, output_folder_path)

    print(output_paths_dict)

if __name__ == "__main__":
    main()
