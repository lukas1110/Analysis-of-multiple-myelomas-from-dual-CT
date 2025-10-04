from radiomics import featureextractor
import SimpleITK as Sitk
import pandas as pd
import numpy as np


# Vertebra names mapping (25 vertebrae)
vertebra_names = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)]        # L1–L5
)

# Initialize default feature extractor with setting you set
features_extractor = featureextractor.RadiomicsFeatureExtractor()
features_extractor.settings['additionalInfo'] = False   # without useless info

# Disable diagnostics
features_extractor.disableAllFeatures()

# Enable features you need:
features_extractor.enableFeatureClassByName("firstorder")
features_extractor.enableFeatureClassByName("glcm")
features_extractor.enableFeatureClassByName("glrlm")
features_extractor.enableFeatureClassByName("glszm")
features_extractor.enableFeatureClassByName("gldm")
features_extractor.enableFeatureClassByName("ngtdm")
features_extractor.enableFeatureClassByName("shape")


# Whole spine features
def radiomics_spine_features(ct_img, vertebrae_img, lesions_img):

    # Load images as numpy ndarray
    vertebrae_img_data = Sitk.GetArrayFromImage(vertebrae_img)
    lesions_img_data = Sitk.GetArrayFromImage(lesions_img)

    # Count number of lesions inside whole spine
    n_lesions_spine = len(np.unique(lesions_img_data[(vertebrae_img_data > 0) & (lesions_img_data > 0)]))

    # Define segmentation masks for lesions and vertebrae without lesions (binary 0/1)
    vertebrae_img_data = ((vertebrae_img_data > 0) & (lesions_img_data == 0)).astype(np.uint8)
    lesions_img_data = (lesions_img_data > 0).astype(np.uint8)

    # Convert to SimpleITK images
    vertebrae_mask = Sitk.GetImageFromArray(vertebrae_img_data)
    lesions_mask = Sitk.GetImageFromArray(lesions_img_data)

    # Copy spatial information from CT
    vertebrae_mask.CopyInformation(ct_img)
    lesions_mask.CopyInformation(ct_img)

    # --- Vertebrae without lesions ---
    v_entry = {}
    v_features = features_extractor.execute(ct_img, vertebrae_mask)
    v_entry.update(v_features)
    pd.DataFrame([v_entry]).to_csv("radiomics_spine_vertebrae_features.csv", index=False)

    # --- All lesions in spine ---
    l_entry = {"n_lesions": n_lesions_spine}
    l_features = features_extractor.execute(ct_img, lesions_mask)
    l_entry.update(l_features)
    pd.DataFrame([l_entry]).to_csv("radiomics_spine_lesions_features.csv", index=False)


# Individual vertebrae features
def radiomics_vertebrae_features(ct_img, vertebrae_img, lesions_img):

    vertebrae_data = Sitk.GetArrayFromImage(vertebrae_img)
    lesions_data = Sitk.GetArrayFromImage(lesions_img)

    vertebrae_results = []
    lesion_results = []

    for v_label in np.unique(vertebrae_data):
        if v_label == 0:
            continue

        single_vertebra_data = (vertebrae_data == v_label)
        single_vertebra_healthy_data = (single_vertebra_data & (lesions_data == 0))
        single_vertebra_lesions_data = (single_vertebra_data & (lesions_data > 0))

        # Number of lesions in single vertebra
        n_lesions = len(np.unique(lesions_data[single_vertebra_lesions_data]))

        # Convert to SimpleITK images
        single_vertebra_healthy_mask = Sitk.GetImageFromArray(single_vertebra_healthy_data.astype(np.uint8))
        single_vertebra_lesions_mask = Sitk.GetImageFromArray(single_vertebra_lesions_data.astype(np.uint8))

        # Copy spatial information
        single_vertebra_healthy_mask.CopyInformation(ct_img)
        single_vertebra_lesions_mask.CopyInformation(ct_img)

        # Radiomics features for healthy vertebra
        v_entry = {
            "vertebra_id": int(v_label),
            "vertebra_name": vertebra_names[int(v_label) - 1],
            "n_lesions": n_lesions
        }

        v_features = features_extractor.execute(ct_img, single_vertebra_healthy_mask)
        v_entry.update(v_features)
        vertebrae_results.append(v_entry)

        if n_lesions > 0:
            # Radiomics features for lesions
            lesion_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": vertebra_names[int(v_label) - 1],
                "n_lesions": n_lesions
            }

            l_features = features_extractor.execute(ct_img, single_vertebra_lesions_mask)
            lesion_entry.update(l_features)
            lesion_results.append(lesion_entry)

    # Save results
    pd.DataFrame(vertebrae_results).to_csv("radiomics_vertebrae_features.csv", index=False)
    pd.DataFrame(lesion_results).to_csv("radiomics_lesions_features.csv", index=False)


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
