from compute_features import *
import pandas as pd


def create_dfs(ct_image, vertebrae_mask, lesions_mask):
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
        v_region = ct_image[vertebrae_mask == v_label]

        # Number of lesion in vertebra
        # noinspection PyUnresolvedReferences
        n_lesions = len(np.unique(lesions_mask_data[(vertebrae_mask_data == v_label)
                                                    & (lesions_mask_data > 0)]))

        # Some basic features for vertebrae
        v_entry = {
            "vertebra_id": int(v_label),
            "vertebra_name": vertebra_names[int(v_label) - 1],
            "n_lesions": n_lesions,
            "mean": float(np.mean(v_region)),
            "std": float(np.std(v_region)),
            "min": float(np.min(v_region)),
            "max": float(np.max(v_region))
        }
        # compute texture features for vertebra region
        v_entry.update(compute_glcm_features_slicewise(vertebrae_mask == v_label, ct_image))
        vertebrae_results.append(v_entry)

        if n_lesions > 0:
            # All lesion voxel for specific vertebra
            lesion_voxels = ct_image[(vertebrae_mask == v_label) & (lesions_mask > 0)]
            lesion_entry = {
                "vertebra_id": int(v_label),
                "vertebra_name": vertebra_names[int(v_label) - 1],
                "n_lesions": n_lesions,
                "mean": float(np.mean(lesion_voxels)),
                "std": float(np.std(lesion_voxels)),
                "min": float(np.min(lesion_voxels)),
                "max": float(np.max(lesion_voxels))
            }
            lesion_entry.update(compute_glcm_features_slicewise(lesions_mask, ct_image))
            lesion_results.append(lesion_entry)

    df_vertebrae = pd.DataFrame(vertebrae_results)
    df_lesions = pd.DataFrame(lesion_results)

    df_vertebrae.to_csv("vertebrae_features01.txt", index=False, sep="\t")  # tab-separated
    df_lesions.to_csv("lesion_features01.txt", index=False, sep="\t")  # tab-separated
