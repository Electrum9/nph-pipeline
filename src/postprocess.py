import nibabel as nib
import numpy as np
import os
import skimage
import copy

parent_dir = pathlib.Path(__file__).resolve().parent
assets_dir = parent_dir / pathlib.Path.cwd() / "assets"

test_transformed_path = "/home/shailja/NPH_mni/"

ventr_prob_map = "MNI_ventr_prob_map_image.nii.gz"
sub_prob_map = "MNI_sub_prob_map_image.nii.gz"
cerebellum_prob_map = "MNI_cereb_prob_map_image.nii.gz"
fourth_prob_map = "MNI_4ventr_prob_map_image.nii.gz"
background_prob_map = "MNI_background_prob_map_image.nii.gz"

ventr_prob_img = nib.load(ventr_prob_map)
ventr_prob_img_data = ventr_prob_img.get_fdata()

sub_prob_img = nib.load(sub_prob_map)
sub_prob_img_data = sub_prob_img.get_fdata()

cerebellum_prob_img = nib.load(cerebellum_prob_map)
cerebellum_prob_img_data = cerebellum_prob_img.get_fdata()

fourth_prob_img = nib.load(fourth_prob_map)
fourth_prob_img_data = fourth_prob_img.get_fdata()

background_prob_img = nib.load(background_prob_map)
background_prob_img_data = background_prob_img.get_fdata()

files_list = os.listdir(test_transformed_path)
#for file in files_list:
#    if ".nii" in file:
#        
#        
#        if "post" not in file and "all" not in file:
#img = nib.load(test_transformed_path+file)


#print(test_transformed_path+file.split('.')[0]+'MNI_post_processed.nii.gz')
#nib.save(post_processed, test_transformed_path+file.split('.')[0]+'MNI_post_processed.nii.gz')
#print(np.unique(post_processed_img, return_counts = True))

def correct(segmented):
    img = segmented # TODO: Replace with nifti object
    img_data = np.round(img.get_fdata())
    #print(np.unique(img_data))
    #dic_label = {}
    post_processed_img= copy.deepcopy(img_data)
    ventr_correction = np.where(
                        img_data == 1,
                        1,
                        0)
    for k in range(img_data.shape[2]):
        labeled_image, count = skimage.measure.label(ventr_correction[:,:,k], return_num=True)
        label, label_count = np.unique(labeled_image, return_counts = True)
    #                 print(count)
        
        for i in range(count+1):
            if (label_count[i]) < 600:
                sub_prob_img_data_sum = np.sum(np.where(( labeled_image == i),sub_prob_img_data[:,:,k],0 ))/label_count[i]
                ventr_prob_img_data_sum = np.sum(np.where(( labeled_image == i),ventr_prob_img_data[:,:,k],0 ))/label_count[i] 
                post_processed_img[:,:,k] = np.where((labeled_image == i) & (sub_prob_img_data_sum>=ventr_prob_img_data_sum), 3, post_processed_img[:,:,k])

    post_processed_img = np.where((ventr_prob_img_data >0)&(post_processed_img ==0),1, post_processed_img)
    post_processed = nib.Nifti1Image(np.array(post_processed_img), img.affine, img.header)

    return post_processed
