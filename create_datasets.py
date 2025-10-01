from extract_features import *
import pandas as pd


# Vertebra names mapping (25 vertebrae)
vertebra_names = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)]        # L1–L5
)

def create_spine_features_dfs(ct_img, vertebrae_img, lesions_img):

    # Get images data
    ct_image_data = ct_img.get_image_data()
    vertebrae_mask_data = vertebrae_img.get_image_data()
    lesions_mask_data = lesions_img.get_image_data()

    # ---------------- Vertebrae without lesions ----------------
    vertebra_voxels = ct_image_data[(vertebrae_mask_data > 0) & (lesions_mask_data == 0)]

    v_entry = {"region": "vertebrae_no_lesions"}
    v_entry.update(extract_intensity_features(vertebra_voxels))
    v_entry.update(extract_glcm_features((vertebrae_mask_data > 0) & (lesions_mask_data == 0), ct_image_data))
    v_entry.update(extract_shape_features((vertebrae_mask_data > 0) & (lesions_mask_data == 0),
                                          spacing=ct_img.get_image_spacing()))

    df_spine_vertebrae = pd.DataFrame([v_entry])
    df_spine_vertebrae.to_csv("spine_vertebrae_features.csv", index=False)

    # ---------------- Lesions inside vertebrae ----------------
    lesion_voxels = ct_image_data[(vertebrae_mask_data > 0) & (lesions_mask_data > 0)]

    # count number of unique lesion labels inside vertebrae
    n_lesions_spine = len(np.unique(lesions_mask_data[(vertebrae_mask_data > 0) & (lesions_mask_data > 0)]))

    l_entry = {
        "region": "lesions",
        "n_lesions": n_lesions_spine
    }

    l_entry.update(extract_intensity_features(lesion_voxels))
    l_entry.update(extract_glcm_features((vertebrae_mask_data > 0) & (lesions_mask_data > 0), ct_image_data))
    l_entry.update(extract_shape_features((vertebrae_mask_data > 0) & (lesions_mask_data > 0),
                                          spacing=ct_img.get_image_spacing()))

    df_spine_lesions = pd.DataFrame([l_entry])
    df_spine_lesions.to_csv("spine_lesions_features.csv", index=False)


def create_vertebra_features_dfs(ct_img, vertebrae_img, lesions_img):

    # Get images data
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


def create_individual_lesion_features_dfs(ct_img, vertebrae_img, lesions_img):

    # Get images data
    ct_image_data = ct_img.get_image_data()
    vertebrae_mask_data = vertebrae_img.get_image_data()
    lesions_mask_data = lesions_img.get_image_data()

    lesion_results = []

    for v_label in np.unique(vertebrae_mask_data):
        if v_label == 0:
            continue

        # Get all lesion labels inside this vertebra
        lesion_labels = np.unique(lesions_mask_data[vertebrae_mask_data == v_label])
        lesion_labels = lesion_labels[lesion_labels > 0]  # remove background

        for l_label in lesion_labels:
            # mask for this specific lesion in this vertebra
            lesion_mask = (lesions_mask_data == l_label) & (vertebrae_mask_data == v_label)
            lesion_voxels = ct_image_data[lesion_mask]

            # entry for this lesion
            lesion_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": vertebra_names[int(v_label) - 1],
                "lesion_id": int(l_label)
            }
            # intensity features
            lesion_entry.update(extract_intensity_features(lesion_voxels))
            # texture features
            lesion_entry.update(extract_glcm_features(lesion_mask, ct_image_data))
            # shape features
            lesion_entry.update(extract_shape_features(lesion_mask,
                                                       spacing=ct_img.get_image_spacing()))

            lesion_results.append(lesion_entry)

    # Save to dataframe
    df_lesions = pd.DataFrame(lesion_results)
    df_lesions.to_csv("individual_lesion_features.csv", index=False)
