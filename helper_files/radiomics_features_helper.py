import numpy as np
import pandas as pd
import SimpleITK as Sitk
from radiomics import featureextractor

# Vertebra names mapping (25 vertebrae)
VERTEBRA_NAMES = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)]        # L1–L5
)

# Initialize default feature extractor with setting you set
features_extractor = featureextractor.RadiomicsFeatureExtractor()
features_extractor.settings['additionalInfo'] = False   # without useless info

# Disable diagnostics
features_extractor.disableAllFeatures()

# Enable features you need
features_set = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"]
for feature in features_set:
    features_extractor.enableFeatureClassByName(feature)


# Whole spine features
def radiomics_spine_features(ct_img, vertebrae_img, lesions_img):

    # Load images as numpy ndarray
    vertebrae_data = Sitk.GetArrayFromImage(vertebrae_img)
    lesions_data = Sitk.GetArrayFromImage(lesions_img)

    v_result = []
    l_result = []

    # Count number of lesions inside whole spine
    n_lesions_spine = len(np.unique(lesions_data[(vertebrae_data > 0) & (lesions_data > 0)]))

    # Define segmentation masks for lesions and vertebrae without lesions (binary 0/1)
    healthy_vertebrae_img_data = (vertebrae_data > 0) & (lesions_data == 0)
    lesions_data = (vertebrae_data > 0) & (lesions_data > 0)

    #Convert to SimpleITK images
    healthy_vertebrae_mask = Sitk.GetImageFromArray(healthy_vertebrae_img_data.astype(np.uint8))
    lesions_mask = Sitk.GetImageFromArray(lesions_data.astype(np.uint8))

    # Copy spatial information from CT
    healthy_vertebrae_mask.CopyInformation(ct_img)
    lesions_mask.CopyInformation(ct_img)

    # --- Vertebrae without lesions ---
    v_entry = {}
    v_features = features_extractor.execute(ct_img, healthy_vertebrae_mask)
    v_entry.update(v_features)
    v_result.append(v_entry)
    pd.DataFrame(v_result).to_csv("radiomics_spine_vertebrae_features.csv", index=False)

    # --- All lesions in spine ---
    l_entry = {"n_lesions": n_lesions_spine}
    l_features = features_extractor.execute(ct_img, lesions_mask)
    l_entry.update(l_features)
    l_result.append(l_entry)
    pd.DataFrame(l_result).to_csv("radiomics_spine_lesions_features.csv", index=False)


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
            "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
            "n_lesions": n_lesions
        }

        v_features = features_extractor.execute(ct_img, single_vertebra_healthy_mask)
        v_entry.update(v_features)
        vertebrae_results.append(v_entry)

        if n_lesions > 0:
            # Radiomics features for lesions
            l_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
                "n_lesions": n_lesions
            }

            l_features = features_extractor.execute(ct_img, single_vertebra_lesions_mask)
            l_entry.update(l_features)
            lesion_results.append(l_entry)

    # Save results
    pd.DataFrame(vertebrae_results).to_csv("radiomics_vertebrae_features.csv", index=False)
    pd.DataFrame(lesion_results).to_csv("radiomics_lesions_features.csv", index=False)


# Individual lesion features
def radiomics_individual_lesion_features(ct_img, vertebrae_img, lesions_img):

    vertebrae_data = Sitk.GetArrayFromImage(vertebrae_img)
    lesions_data = Sitk.GetArrayFromImage(lesions_img)

    results_list = []

    for v_label in np.unique(vertebrae_data):
        if v_label == 0:
            continue

        # Get lesion labels inside specific vertebra
        lesions_labels = np.unique(lesions_data[vertebrae_data == v_label])
        lesions_labels = lesions_labels[lesions_labels > 0]

        for l_label in lesions_labels:
            # Mask for specific lesion
            lesion_mask_data = (lesions_data == l_label) & (vertebrae_data == v_label)

            # Convert to SimpleITK image
            lesion_mask = Sitk.GetImageFromArray(lesion_mask_data.astype(np.uint8))

            # Copy spatial information
            lesion_mask.CopyInformation(ct_img)

            l_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
                "lesion_id": int(l_label)
            }

            l_features = features_extractor.execute(ct_img, lesion_mask)
            l_entry.update(l_features)
            results_list.append(l_entry)

    df_lesions = pd.DataFrame(results_list)
    df_lesions.to_csv("radiomics_individual_lesion_features.csv", index=False)
