from extract_features import *
import pandas as pd


# Defining vertebra names mapping (25 vertebrae)
vertebra_names = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)]        # L1–L5
)

def create_features_dfs(ct_img, vertebrae_img, lesions_img):

    ct_image_data = ct_img.get_image_data()
    vertebrae_mask_data = vertebrae_img.get_image_data()
    lesions_mask_data = lesions_img.get_image_data()

    vertebrae_results = []
    lesion_results = []

    for v_label in np.unique(vertebrae_mask_data):
        if v_label == 0:
            continue

        # Vertebrae voxels
        vertebra_voxels = ct_image_data[(vertebrae_mask_data == v_label) & (lesions_mask_data == 0)]

        # Number of lesion in vertebra
        n_lesions = len(np.unique(lesions_mask_data[(vertebrae_mask_data == v_label) & (lesions_mask_data > 0)]))

        # Some basic features for vertebrae
        v_entry = {
            "vertebra_id": int(v_label),
            "vertebra_name": vertebra_names[int(v_label) - 1],
            "n_lesions": n_lesions
        }
        # Compute intensity features for vertebra region
        v_entry.update(extract_intensity_features(vertebra_voxels))
        # Compute texture features for vertebra region
        v_entry.update(extract_glcm_features((vertebrae_mask_data == v_label) & (lesions_mask_data == 0), ct_image_data))
        # Compute shape features for vertebra region
        v_entry.update(extract_shape_features((vertebrae_mask_data == v_label) & (lesions_mask_data == 0),
                                              spacing=ct_img.get_image_spacing()))

        vertebrae_results.append(v_entry)

        if n_lesions > 0:
            # All lesion voxel for specific vertebra
            lesion_voxels = ct_image_data[(vertebrae_mask_data == v_label) & (lesions_mask_data > 0)]

            # Some basic features for lesions
            lesion_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": vertebra_names[int(v_label) - 1],
                "n_lesions": n_lesions
            }
            # Compute intensity features for lesions region
            lesion_entry.update(extract_intensity_features(lesion_voxels))
            # Compute texture features for lesions region
            lesion_entry.update(extract_glcm_features((vertebrae_mask_data == v_label) & (lesions_mask_data > 0), ct_image_data))
            # Compute shape features for lesions region
            lesion_entry.update(extract_shape_features((vertebrae_mask_data == v_label) & (lesions_mask_data > 0),
                                                       spacing=ct_img.get_image_spacing()))

            lesion_results.append(lesion_entry)

    df_vertebrae = pd.DataFrame(vertebrae_results)
    df_lesions = pd.DataFrame(lesion_results)

    df_vertebrae.to_csv("vertebrae_features.csv", index=False)
    df_lesions.to_csv("lesion_features.csv", index=False)
