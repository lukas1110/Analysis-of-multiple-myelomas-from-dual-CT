from load_image import NiftiFile
from create_datasets import *
from visualization import *


# Path to nifti files
image_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
lesions_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"
vertebrae_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Spine_labels\myel_001_spine_seg_nnUNet_cor.nii.gz"

# Load the images as object from nifti file
ct_img = NiftiFile(image_path)
lesions_img = NiftiFile(lesions_mask_path)
vertebrae_img = NiftiFile(vertebrae_mask_path)

# Datasets
# create_spine_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_vertebra_features_dfs(ct_img, vertebrae_img, lesions_img)
create_individual_lesion_features_dfs(ct_img, vertebrae_img, lesions_img)



# ----------------------------------------------------VISUALIZATION----------------------------------------------------
# Load the images as numpy array
# ct_image_data = ct_img.get_image_data()
# lesions_mask_data = lesions_img.get_image_data()
# vertebrae_mask_data = vertebrae_img.get_image_data()

# Vertebrae or lesions mask
# lesion_voxels = (vertebrae_mask_data == 22) & (lesions_mask_data > 0)
# vertebra_voxels = (vertebrae_mask_data == 14) & (lesions_mask_data == 0)

# Visualizations
# show_individual_slices(ct_image_data)
# show_slices_scroll(ct_image_data, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image_data, vertebra_voxels, axis=0, pause_time=0.01)
# show_slices_scroll_range_with_mask(ct_image_data, vertebra_voxels, start=150, end=300)
