import os
import numpy as np
import pandas as pd
import SimpleITK as Sitk
from radiomics import featureextractor


# Vertebra names mapping (25 vertebrae)
VERTEBRA_NAMES = (
    ["C" + str(i) for i in range(1, 8)] +      # C1–C7
    ["T" + str(i) for i in range(1, 13)] +     # T1–T12
    ["L" + str(i) for i in range(1, 6)] +      # L1–L5
    ["S" + str(i) for i in range(1, 6)]        # S1–S5 (optional)
)

class RadiomicsFeatures:
    """
    Class for extracting radiomics features from CT images of the spine.
    It computes features for:
        1. The entire spine (healthy vertebrae and all lesions combined)
        2. Each vertebra separately (healthy part and its lesions)
        3. Each individual lesion separately

    Attributes:
        ct_img (SimpleITK.Image): CT image.
        vertebrae_data (np.ndarray): Label mask of vertebrae (1–24 labels, 0 = background).
        lesions_data (np.ndarray): Label mask of lesions (unique lesion labels, 0 = background).
        features_extractor (RadiomicsFeatureExtractor): Configured PyRadiomics extractor.
    """
    def __init__(self, ct_img, vertebrae_img, lesions_img):
        """
        Initialize with CT image and segmentation masks.
        """
        self.ct_img = ct_img
        self.vertebrae_data = Sitk.GetArrayFromImage(vertebrae_img)
        self.lesions_data = Sitk.GetArrayFromImage(lesions_img)

        # Initialize and configure the PyRadiomics feature extractor
        self.features_extractor = self._set_features_extractor(
            featureextractor.RadiomicsFeatureExtractor())

    # ------------------------------
    # INTERNAL CONFIGURATION METHODS
    # ------------------------------

    @staticmethod
    def _set_features_extractor(features_extractor):
        """
        Configure a RadiomicsFeatureExtractor instance with selected feature classes.
        Disables all features first, then enables the major ones manually.

        param: features_extractor (RadiomicsFeatureExtractor): RadiomicsFeatureExtractor instance.
        returns: RadiomicsFeatureExtractor instance.
        """
        features_extractor.settings['additionalInfo'] = False   # turn off useless info
        features_extractor.disableAllFeatures()

        # Set another image type (Original + Gradient)
        features_extractor.enableImageTypeByName("Gradient")

        # Enable radiomics feature groups
        features_set = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"]
        for feature in features_set:
            features_extractor.enableFeatureClassByName(feature)
        return features_extractor

    def _convert_array_to_sitk_image(self, image_data):
        """
        Convert a NumPy mask (binary or labeled) to a SimpleITK image
        while copying spatial metadata (spacing, direction, origin) from the CT.

        param: image_data: numpy array.
        returns: SimpleITK image.
        """
        sitk_image = Sitk.GetImageFromArray(image_data.astype(np.uint8))
        sitk_image.CopyInformation(self.ct_img)
        return sitk_image

    @staticmethod
    def _save_csv(result_to_save, csv_name, path_to_save: str=None):
        """
        Save list of dictionaries to a CSV file.

        param: result_to_save (list): List of dictionaries.
        param: csv_name (str): Name of CSV file.
        optional: path_to_save (str): Path to save CSV file.
        """
        if path_to_save is not None:
            pd.DataFrame(result_to_save).to_csv(os.path.join(path_to_save, csv_name), index=False)
        else:
            pd.DataFrame(result_to_save).to_csv(csv_name, index=False)

    # --------------------------
    # FEATURE EXTRACTION METHODS
    # --------------------------

    def extract_radiomics_spine_features(self):
        """
        Extract radiomics features for:
            - All healthy vertebrae combined (vertebra mask without lesions)
            - All lesions combined (within vertebrae)

        Produces:
            - radiomics_spine_vertebrae_features.csv
            - radiomics_spine_lesions_features.csv
        """
        # Predefine result lists
        v_result, l_result = [], []

        # Count number of lesions inside whole spine
        n_lesions_spine = len(np.unique(self.lesions_data[(self.vertebrae_data > 0) & (self.lesions_data > 0)]))

        # Define segmentation masks for lesions and vertebrae without lesions (binary 0/1)
        healthy_vertebrae_img_data = (self.vertebrae_data > 0) & (self.lesions_data == 0)
        lesions_data = (self.vertebrae_data > 0) & (self.lesions_data > 0)

        # --- Vertebrae without lesions ---
        v_entry = {}
        v_features = self.features_extractor.execute(
            self.ct_img, self._convert_array_to_sitk_image(healthy_vertebrae_img_data))
        v_entry.update(v_features)
        v_result.append(v_entry)

        # --- All lesions in spine ---
        l_entry = {"n_lesions": n_lesions_spine}
        l_features = self.features_extractor.execute(
            self.ct_img, self._convert_array_to_sitk_image(lesions_data))
        l_entry.update(l_features)
        l_result.append(l_entry)

        # Save DataFrames to csv files
        self._save_csv(v_result, "radiomics_spine_vertebrae_features.csv")
        self._save_csv(l_result, "radiomics_spine_lesions_features.csv")

    def extract_radiomics_vertebrae_features(self):
        """
        Extract radiomics features for each vertebra separately.
        For each vertebra:
            - Healthy part (no lesions)
            - Lesions within that vertebra (if any)
        Produces:
            - radiomics_vertebrae_features.csv
            - radiomics_lesions_features.csv
        """
        # Predefine result lists
        vertebrae_results, lesion_results = [], []

        for v_label in np.unique(self.vertebrae_data):
            if v_label == 0:
                continue   # skip background

            # Boolean masks for this vertebra
            single_vertebra_data = (self.vertebrae_data == v_label)
            single_vertebra_healthy_data = (single_vertebra_data & (self.lesions_data == 0))
            single_vertebra_lesions_data = (single_vertebra_data & (self.lesions_data > 0))

            # Number of lesions in this vertebra
            n_lesions = len(np.unique(self.lesions_data[single_vertebra_lesions_data]))

            # --- Healthy vertebra features ---
            v_entry = {
                "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
                "n_lesions": n_lesions
            }

            v_features = self.features_extractor.execute(
                self.ct_img, self._convert_array_to_sitk_image(single_vertebra_healthy_data))
            v_entry.update(v_features)
            vertebrae_results.append(v_entry)

            # --- Lesions features (only if present) ---
            if n_lesions > 0:
                l_entry = {
                    "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
                    "n_lesions": n_lesions
                }

                l_features = self.features_extractor.execute(
                    self.ct_img, self._convert_array_to_sitk_image(single_vertebra_lesions_data))
                l_entry.update(l_features)
                lesion_results.append(l_entry)

        # Save DataFrames to csv files
        self._save_csv(vertebrae_results, "radiomics_vertebrae_features.csv")
        self._save_csv(lesion_results, "radiomics_lesions_features.csv")

    def extract_radiomics_individual_lesion_features(self):
        """
        Extract radiomics features for each individual lesion separately.

        Produces:
            - radiomics_individual_lesion_features.csv
        """
        # Predefine result list
        results_list = []

        for v_label in np.unique(self.vertebrae_data):
            if v_label == 0:
                continue   # skip background

            # Get lesion labels inside specific vertebra
            lesions_labels = np.unique(self.lesions_data[self.vertebrae_data == v_label])
            lesions_labels = lesions_labels[lesions_labels > 0]

            for l_label in lesions_labels:
                # Binary mask for this specific lesion
                lesion_mask_data = (self.lesions_data == l_label) & (self.vertebrae_data == v_label)

                l_entry = {
                    "vertebra_name": VERTEBRA_NAMES[int(v_label) - 1],
                }

                l_features = self.features_extractor.execute(
                    self.ct_img, self._convert_array_to_sitk_image(lesion_mask_data))
                l_entry.update(l_features)
                results_list.append(l_entry)

        # Save DataFrame to csv file
        self._save_csv(results_list, "radiomics_individual_lesion_features.csv")
