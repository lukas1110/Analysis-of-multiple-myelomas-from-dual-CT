from load_image import load_nifti_image
from visualization import *


# Path to nifti file
file_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"

img_data = load_nifti_image(file_path)
mask_data = load_nifti_image(mask_path)

print("Data shape:", img_data.shape)
print("Data dtype:", img_data.dtype)
print("Data type:", type(img_data))

show_individual_slices(img_data)
# show_slices_scroll(img_data, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(img_data, mask_data, axis=0, pause_time=0.01)
