from radiomics import featureextractor
import SimpleITK as Sitk
import pandas as pd


# Vertebra names mapping (25 vertebrae)
vertebra_names = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)]        # L1–L5
)

# Initialize default feature extractor
features_extractor = featureextractor.RadiomicsFeatureExtractor()
features_extractor.enableAllFeatures()
features_extractor.disableAllImageTypes()   # only original image, default features

# Whole spine features
def radiomics_spine_features(image_path, vertebrae_path, lesions_path):

    # Load images as SimpleITK objects
    ct_img = Sitk.ReadImage(image_path)
    vertebrae_img = Sitk.ReadImage(vertebrae_path)
    lesions_img = Sitk.ReadImage(lesions_path)

    # Define segmentation masks for vertebrae and lesions
    vertebra_mask = Sitk.Cast(vertebrae_img > 0, Sitk.sitkUInt8)
    lesions_mask = Sitk.Cast(lesions_img > 0, Sitk.sitkUInt8)

    # --- Vertebrae without lesions ---
    healthy_vertebra_mask = vertebra_mask * Sitk.InvertIntensity(lesions_mask, maximum=1)
    v_entry = {"region": "vertebrae_no_lesions"}
    features = features_extractor.execute(ct_img, healthy_vertebra_mask)
    v_entry.update(features)
    df_spine_vertebrae = pd.DataFrame([v_entry])
    df_spine_vertebrae.to_csv("radiomics_spine_vertebrae_features.csv", index=False)

    # --- All lesions ---
    l_entry = {"region": "lesions"}
    features = features_extractor.execute(ct_img, lesions_mask)
    l_entry.update(features)
    df_spine_lesions = pd.DataFrame([l_entry])
    df_spine_lesions.to_csv("radiomics_spine_lesions_features.csv", index=False)


# Individual vertebrae features
# def radiomics_vertebrae_features(ct_img, vertebrae_img, lesions_img):
#     vertebrae_results = []
#     lesion_results = []
#
#     vertebrae_data = sitk.GetArrayFromImage(vertebrae_img)
#     lesions_data = sitk.GetArrayFromImage(lesions_img)
#
#     for v_label in np.unique(vertebrae_data):
#         if v_label == 0:
#             continue
#
#         # --- Vertebra voxels without lesions ---
#         v_mask_array = (vertebrae_data == v_label) & (lesions_data == 0)
#
#         v_mask_sitk = sitk.GetImageFromArray(v_mask_array.astype(np.uint8))
#         v_mask_sitk.CopyInformation(ct_img)
#
#         v_entry = {
#             "vertebra_id": int(v_label),
#             "vertebra_name": vertebra_names[int(v_label) - 1]
#         }
#
#         features = features_extractor.execute(ct_img, v_mask_sitk)
#         v_entry.update(features)
#         vertebrae_results.append(v_entry)
#
#         # --- Lesions inside vertebra ---
#         l_mask_array = (vertebrae_data == v_label) & (lesions_data > 0)
#
#         l_mask_sitk = sitk.GetImageFromArray(l_mask_array.astype(np.uint8))
#         l_mask_sitk.CopyInformation(ct_img)
#
#         lesion_entry = {
#             "vertebra_id": int(v_label),
#             "vertebra_name": vertebra_names[int(v_label) - 1]
#         }
#
#         features = features_extractor.execute(ct_img, l_mask_sitk)
#         lesion_entry.update(features)
#         lesion_results.append(lesion_entry)
#
#     # Save to CSV
#     df_vertebrae = pd.DataFrame(vertebrae_results)
#     df_lesions = pd.DataFrame(lesion_results)
#
#     df_vertebrae.to_csv("radiomics_vertebrae_features.csv", index=False)
#     df_lesions.to_csv("radiomics_lesions_features.csv", index=False)
#
#
# # Individual lesion features
# def radiomics_individual_lesion_features(ct_img, vertebrae_img, lesions_img):
#
#     results_list = []
#
#     vertebrae_data = sitk.GetArrayFromImage(vertebrae_img)
#     lesions_data = sitk.GetArrayFromImage(lesions_img)
#
#     for v_label in np.unique(vertebrae_data):
#         if v_label == 0:
#             continue
#
#         # get lesion labels inside this vertebra
#         lesion_labels = np.unique(lesions_data[vertebrae_data == v_label])
#         lesion_labels = lesion_labels[lesion_labels > 0]
#
#         for l_label in lesion_labels:
#             # mask for this lesion
#             l_mask_array = (lesions_data == l_label) & (vertebrae_data == v_label)
#
#             l_mask_sitk = sitk.GetImageFromArray(l_mask_array.astype(np.uint8))
#             l_mask_sitk.CopyInformation(ct_img)
#
#             entry = {
#                 "vertebra_id": int(v_label),
#                 "vertebra_name": vertebra_names[int(v_label)-1],
#                 "lesion_id": int(l_label)
#             }
#
#             features = features_extractor.execute(ct_img, l_mask_sitk)
#             entry.update(features)
#             results_list.append(entry)
#
#     df_lesions = pd.DataFrame(results_list)
#     df_lesions.to_csv("radiomics_individual_lesion_features.csv", index=False)
