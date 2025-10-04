from load_image import NiftiFile
from create_datasets import *
from visualization import *
from radiomics_features import *
import SimpleITK as Sitk
from radiomics import featureextractor


# ----------------------------------------------------PATHS TO FILES----------------------------------------------------
image_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
vertebrae_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Spine_labels\myel_001_spine_seg_nnUNet_cor.nii.gz"
lesions_mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"





# ----------------------------------------------------NIBABEL OBJECTS---------------------------------------------------
# ct_img = NiftiFile(image_path)
# vertebrae_img = NiftiFile(vertebrae_mask_path)
# lesions_img = NiftiFile(lesions_mask_path)

# ---------------------------------------------------SIMPLE-ITK OBJECTS-------------------------------------------------
ct_img = Sitk.ReadImage(image_path)
vertebrae_img = Sitk.ReadImage(vertebrae_mask_path)
lesions_img = Sitk.ReadImage(lesions_mask_path)





# ----------------------------------------------CONVERT NIBABEL TO NUMPY ARRAY------------------------------------------
# ct_image_data = ct_img.get_image_data()
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





# ----------------------------------------------------VISUALIZATION----------------------------------------------------
# show_individual_slices(ct_image_data)
# show_slices_scroll(ct_image_data, axis=0, pause_time=0.01)
# show_slices_scroll_with_mask(ct_image_data, vertebra_voxels, axis=0, pause_time=0.01)
# show_slices_scroll_range_with_mask(ct_image_data, single_vertebra_lesions_data, axis=2 ,start=250, end=300)





# --------------------------------------CREATE DATASET USING NIBABEL AND NORMAL APPROACH--------------------------------
# create_spine_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_vertebra_features_dfs(ct_img, vertebrae_img, lesions_img)
# create_individual_lesion_features_dfs(ct_img, vertebrae_img, lesions_img)

# ----------------------------------------CREATE DATASET USING SIMPLE-ITK AND RADIOMICS---------------------------------
# radiomics_spine_features(ct_img, vertebrae_img, lesions_img)
# radiomics_vertebrae_features(ct_img, vertebrae_img, lesions_img)





# ---------------------------------------------RADIOMICS FEATURES EXTRACTOR---------------------------------------------
# # Initialize default feature extractor with setting you set
# features_extractor = featureextractor.RadiomicsFeatureExtractor()
# features_extractor.settings['additionalInfo'] = False   # without useless info
#
# # Disable diagnostics
# features_extractor.disableAllFeatures()
#
# # Enable features you need:
# features_extractor.enableFeatureClassByName("firstorder")
# features_extractor.enableFeatureClassByName("glcm")
# features_extractor.enableFeatureClassByName("glrlm")
# features_extractor.enableFeatureClassByName("glszm")
# features_extractor.enableFeatureClassByName("gldm")
# features_extractor.enableFeatureClassByName("ngtdm")
# features_extractor.enableFeatureClassByName("shape")
#
# # Print feature extractor settings
# for key in features_extractor.settings:
#     print(key, ":", features_extractor.settings[key])
# print()
# print(features_extractor.enabledFeatures)
# print()
# print(features_extractor.enabledImagetypes)
