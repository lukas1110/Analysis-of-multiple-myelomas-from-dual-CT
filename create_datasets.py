from extract_features import *
import pandas as pd


def create_features_dfs(ct_image, vertebrae_mask, lesions_mask):
    # Defining vertebra names mapping (25 vertebrae)
    vertebra_names = (
        ["C" + str(i) for i in range(1, 8)] +      # C1–C7
        ["T" + str(i) for i in range(1, 13)] +     # T1–T12
        ["L" + str(i) for i in range(1, 6)]        # L1–L5
    )

    vertebrae_results = []
    lesion_results = []

    for v_label in np.unique(vertebrae_mask):
        if v_label == 0:
            continue

        # Vertebrae voxels
        vertebra_voxels = ct_image[(vertebrae_mask == v_label) & (lesions_mask == 0)]

        # Number of lesion in vertebra
        n_lesions = len(np.unique(lesions_mask[(vertebrae_mask == v_label) & (lesions_mask > 0)]))

        # Some basic features for vertebrae
        v_entry = {
            "vertebra_id": int(v_label),
            "vertebra_name": vertebra_names[int(v_label) - 1],
            "n_lesions": n_lesions
        }
        # Compute intensity features for vertebra region
        v_entry.update(compute_intensity_features(vertebra_voxels))
        # Compute texture features for vertebra region
        v_entry.update(compute_glcm_features_slicewise((vertebrae_mask == v_label) & (lesions_mask == 0), ct_image))
        # Compute shape features for vertebra region
        v_entry.update(compute_shape_features((vertebrae_mask == v_label) & (lesions_mask == 0), spacing=(0.9, 0.9, 0.9)))

        vertebrae_results.append(v_entry)

        if n_lesions > 0:
            # All lesion voxel for specific vertebra
            lesion_voxels = ct_image[(vertebrae_mask == v_label) & (lesions_mask > 0)]

            # Some basic features for lesions
            lesion_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": vertebra_names[int(v_label) - 1],
                "n_lesions": n_lesions
            }
            # Compute intensity features for lesions region
            lesion_entry.update(compute_intensity_features(lesion_voxels))
            # Compute texture features for lesions region
            lesion_entry.update(compute_glcm_features_slicewise((vertebrae_mask == v_label) & (lesions_mask > 0), ct_image))
            # Compute shape features for lesions region
            lesion_entry.update(compute_shape_features((vertebrae_mask == v_label) & (lesions_mask > 0), spacing=(0.9, 0.9, 0.9)))

            lesion_results.append(lesion_entry)

    df_vertebrae = pd.DataFrame(vertebrae_results)
    df_lesions = pd.DataFrame(lesion_results)

    df_vertebrae.to_csv("vertebrae_features.csv", index=False)
    df_lesions.to_csv("lesion_features.csv", index=False)
