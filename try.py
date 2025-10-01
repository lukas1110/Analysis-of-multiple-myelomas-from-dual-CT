from load_image import NiftiFile
from create_datasets import *
from visualization import *
from radiomics_features import *
import SimpleITK as Sitk


# Path to files
image_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
vertebrae_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Spine_labels\myel_001_spine_seg_nnUNet_cor.nii.gz"
lesions_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"



# Load the images as Nibabel object
ct_img = NiftiFile(image_path)
vertebrae_img = NiftiFile(vertebrae_mask_path)
lesions_img = NiftiFile(lesions_mask_path)

# Load the images as Sitk object
# ct_img = Sitk.ReadImage(image_path)
# vertebrae_img = Sitk.ReadImage(vertebrae_mask_path)
# lesions_img = Sitk.ReadImage(lesions_mask_path)



# Load the images as numpy array
ct_image_data = ct_img.get_image_data()
vertebrae_mask_data = vertebrae_img.get_image_data()
lesions_mask_data = lesions_img.get_image_data()

# Load the images as numpy array
# ct_image_data = Sitk.GetArrayFromImage(ct_img)
# vertebrae_image_data = Sitk.GetArrayFromImage(vertebrae_img)
# lesions_image_data = Sitk.GetArrayFromImage(lesions_img)
# healthy_vertebra_data = Sitk.GetArrayFromImage(healthy_vertebra_mask)



# Datasets
# create_spine_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_vertebra_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_individual_lesion_features_dfs(ct_img, vertebrae_img, lesions_img)

# Datasets using radiomics
# radiomics_spine_features(image_path, vertebrae_mask_path, lesions_mask_path)



# Vertebrae or lesions mask
# vertebra_voxels = (vertebrae_mask_data == 14) & (lesions_mask_data == 0)
# lesion_voxels = (vertebrae_mask_data == 22) & (lesions_mask_data > 0)

# Vertebrae or lesions mask
# vertebra_mask = Sitk.Cast(vertebrae_img > 0, Sitk.sitkUInt8)
# lesions_mask = Sitk.Cast(lesions_img > 0, Sitk.sitkUInt8)
# healthy_vertebra_mask = vertebra_mask * Sitk.InvertIntensity(lesions_mask, maximum=1)



# ----------------------------------------------------VISUALIZATION----------------------------------------------------
# show_individual_slices(ct_image_data)
# show_slices_scroll(ct_image_data, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image_data, vertebra_voxels, axis=0, pause_time=0.01)
# show_slices_scroll_range_with_mask(ct_image_data, vertebra_voxels ,start=150, end=300)
