import nibabel as nib
import numpy as np
import os
import skimage
import copy
from assets import prob_map

ventr_prob_img_data      = nib.load(prob_map['ventr']).get_fdata()
sub_prob_img_data        = nib.load(prob_map['sub']).get_fdata()
cerebellum_prob_img_data = nib.load(prob_map['cereb']).get_fdata()
fourth_prob_img_data     = nib.load(prob_map['4ventr']).get_fdata()
background_prob_img_data = nib.load(prob_map['background']).get_fdata()


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
