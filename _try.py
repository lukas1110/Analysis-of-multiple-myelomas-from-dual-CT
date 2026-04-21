from radiomics_features import RadiomicsFeatures
import SimpleITK as Sitk
from helper_files.visualization import *
from helper_files.load_nifti_image import NiftiFile
from radiomics import featureextractor
import pandas as pd
import numpy as np

# ----------------------------------------------------PATHS TO FILES----------------------------------------------------
image_path = r"E:\DATA_Healthy_and_Myelom\DATA_Healthy\Healthy_001\healthy_001_konv.nii.gz"
# vertebrae_mask_path = r"E:\DATA_Myelomy\Myel_043\myel_043_spine_seg.nii.gz"
# lesions_mask_path = r"E:\DATA_Myelomy\Myel_043\Myel_043_lesions_seg.nii.gz"





# ----------------------------------------------------NIBABEL OBJECTS---------------------------------------------------
ct_img = NiftiFile(image_path)
# vertebrae_img = NiftiFile(vertebrae_mask_path)
# lesions_img = NiftiFile(lesions_mask_path)

# ---------------------------------------------------SIMPLE-ITK OBJECTS-------------------------------------------------
# ct_img = Sitk.ReadImage(image_path)
# vertebrae_img = Sitk.ReadImage(vertebrae_mask_path)
# lesions_img = Sitk.ReadImage(lesions_mask_path)





# ----------------------------------------------CONVERT NIBABEL TO NUMPY ARRAY------------------------------------------
ct_image_data = ct_img.get_image_data()
# vertebrae_image_data = vertebrae_img.get_image_data()
# lesions_image_data = lesions_img.get_image_data()

# --------------------------------------------CONVERT SIMPLE-ITK TO NUMPY ARRAY-----------------------------------------
# ct_image_data = Sitk.GetArrayFromImage(ct_img)
# vertebrae_image_data = Sitk.GetArrayFromImage(vertebrae_img)
# lesions_image_data = Sitk.GetArrayFromImage(lesions_img)





# --------------------------------CREATE SPECIFIC SEGMENTATION MASKS FOR VISUALIZATION - NIBABEL------------------------
# vertebrae_image_data = (vertebrae_image_data == 22) & (lesions_image_data == 0)
# lesions_image_data = (vertebrae_image_data == 22) & (lesions_image_data > 0)

# ------------------------------CREATE SPECIFIC SEGMENTATION MASKS FOR VISUALIZATION - SIMPLE-ITK-----------------------
# whole spine
# vertebrae_image_data = (vertebrae_image_data > 0) & (lesions_image_data == 0)
# lesions_image_data = (lesions_image_data > 0)


# single vertebra
# single_vertebra_data = (vertebrae_image_data == 14)
# single_vertebra_healthy_data = single_vertebra_data & (lesions_image_data == 0)
# single_vertebra_lesions_data = single_vertebra_data & (lesions_image_data > 0)

# n_lesions = len(np.unique(lesions_image_data[single_vertebra_lesions_data]))
# print(n_lesions)


# single lesion
# lesions_labels = np.unique(lesions_image_data[vertebrae_image_data == 14])
# lesions_labels = lesions_labels[lesions_labels > 0]
# lesion_mask_data = (lesions_image_data == 1088) & (vertebrae_image_data == 14)
# print(lesions_labels)





# ----------------------------------------------------VISUALIZATION----------------------------------------------------
z_min, z_max = 10, 40
y_min, y_max = 250, 280
x_min, x_max = 100, 250

mask = np.zeros_like(ct_image_data, dtype=np.uint8)
mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1

print(np.std(ct_image_data[mask==1], ddof=1))

# show_individual_slices(ct_image_data)
# show_slices_scroll(ct_image_data, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image_data, mask, axis=0, pause_time=0.01)
# show_slices_scroll_range_with_mask(ct_image_data, mask, axis=0 ,start=150, end=200)





# --------------------------------------CREATE DATASET USING NIBABEL AND NORMAL APPROACH--------------------------------
# create_spine_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_vertebra_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_individual_lesion_features_dfs(ct_img, vertebrae_img, lesions_img)

# ----------------------------------------CREATE DATASET USING SIMPLE-ITK AND RADIOMICS---------------------------------
# RadiomicsFeatures(ct_img, vertebrae_img, lesions_img).extract_radiomics_spine_features()
# RadiomicsFeatures(ct_img, vertebrae_img, lesions_img).extract_radiomics_vertebrae_features()
# RadiomicsFeatures(ct_img, vertebrae_img, lesions_img).extract_radiomics_individual_lesion_features()





# ---------------------------------------------RADIOMICS FEATURES EXTRACTOR---------------------------------------------
# # Initialize default feature extractor with setting you set
# features_extractor = featureextractor.RadiomicsFeatureExtractor()
# features_extractor.settings['additionalInfo'] = False   # without useless info
#
# # Disable diagnostics
# features_extractor.disableAllFeatures()
# features_extractor.enableImageTypeByName("Gradient")
#
# # Enable features you need:
# features_set = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"]
#
# for feature in features_set:
#     features_extractor.enableFeatureClassByName(feature)
#
# # Print feature extractor settings
# for key in features_extractor.settings:
#     print(key, ":", features_extractor.settings[key])
# print()
# print(features_extractor.enabledFeatures)
# print()
# print(features_extractor.enabledImagetypes)





# ------------------------------------------------ NEW PARAMETRIC MAP --------------------------------------------------
# c_25 = Sitk.GetArrayFromImage(Sitk.ReadImage(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\myel_001_CaSupp_25.nii.gz"))
# vmi_40 = Sitk.GetArrayFromImage(Sitk.ReadImage(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\myel_001_monoe_40kev.nii.gz"))
# vmi_120 = Sitk.GetArrayFromImage(Sitk.ReadImage(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\myel_001_monoe_120kev.nii.gz"))
# I = Sitk.ReadImage(r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\myel_001_CaSupp_25.nii.gz")
#
# c_25_slice = c_25[:, :, 265]
# vmi_40_slice = vmi_40[:, :, 265]
# vmi_120_slice = vmi_120[:, :, 265]
#
# new_slice = ((vmi_120_slice + vmi_40_slice) - c_25_slice)
#
# plt.subplot(131)
# plt.imshow(c_25_slice, cmap="gray", origin="lower")
# plt.subplot(132)
# plt.imshow(vmi_40_slice, cmap="gray", origin="lower")
# plt.subplot(133)
# plt.imshow(new_slice, cmap="gray", origin="lower")
# plt.title("New slice")
# plt.show()

# img_array = ((vmi_120 + vmi_40) - c_25)
# img = Sitk.GetImageFromArray(img_array)
# img.CopyInformation(I)
# Sitk.WriteImage(img, "image.nii.gz")


