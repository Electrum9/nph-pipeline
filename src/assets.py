import pathlib

# ASSETS LOCATION
parent_dir = pathlib.Path(__file__).resolve().parent
assets_dir = parent_dir / "assets"
model_dir = parent_dir

# REGISTRATION FILES
MNI_152_bone = assets_dir / 'MNI152_T1_1mm_bone.nii.gz'
MNI_152 = assets_dir / 'MNI152_T1_1mm.nii.gz'

# PROBABILITY MAPS
prob_map = {'ventr'      : assets_dir / "MNI_ventr_prob_map_image.nii.gz",      
            'sub'        : assets_dir / "MNI_sub_prob_map_image.nii.gz",        
            'cereb'      : assets_dir / "MNI_cereb_prob_map_image.nii.gz",      
            '4ventr'     : assets_dir / "MNI_4ventr_prob_map_image.nii.gz",    
            'background' : assets_dir / "MNI_background_prob_map_image.nii.gz", 
           }

# MODEL FILES
model = model_dir / "test_bs200.onnx"

# TODO: Delete below!

# REFERENCES

reference_segmented_scan = parent_dir / "reconstructed_ct_rest.nii.gz"
