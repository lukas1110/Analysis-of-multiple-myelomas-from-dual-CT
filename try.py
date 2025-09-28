from load_image import load_nifti_image
from create_features_dfs import create_features_dfs
from visualization import *


# Path to nifti files
image_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
lesions_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"
vertebrae_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Spine_labels\myel_001_spine_seg_nnUNet_cor.nii.gz"

# Load the images to numpy_array
ct_image = load_nifti_image(image_path)
lesions_mask = load_nifti_image(lesions_mask_path)
vertebrae_mask = load_nifti_image(vertebrae_mask_path)

# Vertebrae or lesions mask
# lesion_voxels = (vertebrae_mask == 22) & (lesions_mask > 0)
# vertebra_voxels = (vertebrae_mask == 14) & (lesions_mask == 0)

# Visualizations
# show_individual_slices(ct_image)
# show_slices_scroll(ct_image, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image, vertebra_voxels, axis=0, pause_time=0.01)
# show_slices_scroll_range_with_mask(ct_image, vertebra_voxels, start=150, end=300)

# Datasets
# create_features_dfs(ct_image, vertebrae_mask, lesions_mask)
