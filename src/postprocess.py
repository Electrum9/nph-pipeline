import nibabel as nib
import numpy as np
import os
import skimage
import copy

parent_dir = pathlib.Path(__file__).resolve().parent
assets_dir = parent_dir / "assets"

                                                                      # NAME
probmap_files = [assets_dir / "MNI_ventr_prob_map_image.nii.gz",      # ventr
                 assets_dir / "MNI_sub_prob_map_image.nii.gz",        # sub
                 assets_dir / "MNI_cereb_prob_map_image.nii.gz",      # cereb
                 assets_dir / "MNI_4ventr_prob_map_image.nii.gz",     # 4ventr
                 assets_dir / "MNI_background_prob_map_image.nii.gz", # background
                ]

get_name = lambda f: f.split('_')[1]

prob_map = {get_name(f) : nib.load(f).get_fdata() for f in probmap_files} # dictionary of probability map arrays, for each brain region

ventr_prob_img_data      = prob_map['ventr']
sub_prob_img_data        = prob_map['sub']
cerebellum_prob_img_data = prob_map['cereb']
fourth_prob_img_data     = prob_map['4ventr']
background_prob_img_data = prob_map['background']


def correct(img):
    """ Intakes segmented image and corrects it (post-processing).
    """

    img_data = np.round(img.get_fdata())
    #print(np.unique(img_data))
    post_processed_img = copy.deepcopy(img_data)
    ventr_correction = np.where(
                        img_data == 1,
                        1,
                        0)
    for k in range(img_data.shape[2]):
        labeled_image, count = skimage.measure.label(ventr_correction[:,:,k], return_num=True)
        label, label_count = np.unique(labeled_image, return_counts = True)
        
        for i in range(count+1):
            if (label_count[i]) < 600:
                sub_prob_img_data_sum = np.sum(np.where(( labeled_image == i),sub_prob_img_data[:,:,k],0 ))/label_count[i]
                ventr_prob_img_data_sum = np.sum(np.where(( labeled_image == i),ventr_prob_img_data[:,:,k],0 ))/label_count[i] 
                post_processed_img[:,:,k] = np.where((labeled_image == i) & (sub_prob_img_data_sum>=ventr_prob_img_data_sum), 3, post_processed_img[:,:,k])

    post_processed_img = np.where((ventr_prob_img_data >0)&(post_processed_img ==0),1, post_processed_img)
    post_processed = nib.Nifti1Image(np.array(post_processed_img), img.affine, img.header)

    return post_processed
