from load_image import load_nifti_image
from visualization import *


# Path to nifti files
image_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
lesions_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"
vertebrae_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\Spine_labels\myel_001_spine_seg_nnUNet_cor.nii.gz"

# Load the images to numpy_array
ct_image = load_nifti_image(image_path)
lesions_mask = load_nifti_image(lesions_mask_path)
vertebrae_mask = load_nifti_image(vertebrae_mask_path)


# show_individual_slices(ct_image)
# show_slices_scroll(ct_image, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image, lesions_mask, axis=0, pause_time=0.01)

# single_vertebrae_mask = (spine_mask_data == 19)
# show_slices_scroll_range_with_mask(ct_image, single_vertebrae_mask,
#                                    0, 0.025, 200, 300)
