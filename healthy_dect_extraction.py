import os
import numpy as np
import pandas as pd

import SimpleITK as Sitk
from pathlib import Path

from radiomics import featureextractor


class RadiomicsFeaturesExtraction:
    def __init__(self, ct_img, vertebrae_mask):
        self.ct_img = ct_img
        self.vertebrae_mask = Sitk.GetArrayFromImage(vertebrae_mask)

        # Initialize and configure the PyRadiomics feature extractor
        self.features_extractor = self._set_features_extractor(
            featureextractor.RadiomicsFeatureExtractor())

    @staticmethod
    def _set_features_extractor(features_extractor):
        features_extractor.settings['additionalInfo'] = False   # turn off useless info
        features_extractor.disableAllFeatures()

        # Set another image type (Original + Gradient)
        features_extractor.enableImageTypeByName("Gradient")

        # Enable radiomics feature groups
        features_set = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
        for feature in features_set:
            features_extractor.enableFeatureClassByName(feature)
        return features_extractor

    def _convert_array_to_sitk_image(self, image_data):
        sitk_image = Sitk.GetImageFromArray(image_data.astype(np.uint8))
        sitk_image.CopyInformation(self.ct_img)
        return sitk_image

    @staticmethod
    def _save_csv(result_to_save, csv_name, path_to_save: str=None):
        if path_to_save is not None:
            pd.DataFrame(result_to_save).to_csv(os.path.join(path_to_save, csv_name), index=False)
        else:
            pd.DataFrame(result_to_save).to_csv(csv_name, index=False)

    def extract_radiomics_spine_features(self, csv_name: str=None, path_to_save: str=None):
        v_result = []

        # Define segmentation mask (binary 0/1)
        vertebrae_img_data = self.vertebrae_mask > 0

        v_entry = {}
        try:
            v_features = self.features_extractor.execute(
                self.ct_img, self._convert_array_to_sitk_image(vertebrae_img_data))
        except Exception as e:
            print(f"Skipping spine features for image: {csv_name}: {e}")
            v_features = {}

        v_entry.update(v_features)
        v_result.append(v_entry)

        # Save DataFrames to csv files
        if csv_name is not None:
            self._save_csv(v_result, csv_name=csv_name + "_radiomics_spine_features.csv",
                           path_to_save=path_to_save)
        else:
            self._save_csv(v_result, "radiomics_spine_features.csv",
                           path_to_save=path_to_save)


class ExtractPatientsData:
    def __init__(self, base_folder_path, folder_name):
        self.base_folder_path = base_folder_path
        self.folder_name = folder_name

    def _extract_image_name(self, path):
        image_name = Path(path).with_suffix("").with_suffix("").name
        if image_name.lower().startswith(self.folder_name):
            # Remove prefix
            return "_".join(image_name.split("_")[2:])
        return image_name

    @staticmethod
    def _extract_mask(myel_path):
        image_files_path = []
        mask_file_path = None

        # Loop through each file inside the patient folder
        for file_name in os.listdir(myel_path):
            file_path = os.path.join(myel_path, file_name)

            # Skip if it's not a file or not a .nii.gz file
            if not (os.path.isfile(file_path) and file_name.lower().endswith(".nii.gz")):
                continue

            # Separate images and segmentation masks based on filename
            if "seg" in file_name.lower():
                mask_file_path = file_path
            else:
                image_files_path.append(file_path)

        return image_files_path, mask_file_path

    def _create_dataset(self, image_files_path, mask_file_path, path_to_save: str=None):
        # Load vertebrae mask
        vertebrae_mask = Sitk.ReadImage(mask_file_path)

        # Load individual CT images and create dataset using image and masks
        for image_path in image_files_path:
            # Extract image name
            image_name = self._extract_image_name(image_path)
            print(f"{image_name}")

            # Load images to Sitk objects
            ct_img = Sitk.ReadImage(image_path)

            # Extract features and save them to csv
            radiomics_features = RadiomicsFeaturesExtraction(ct_img, vertebrae_mask)

            if os.path.exists(os.path.join(os.path.dirname(image_path), image_name + "_radiomics_spine_features.csv")):
                print(f"CSV {image_name} for spine already exists, skipping features extraction.")
            else:
                print("Spine level")
                radiomics_features.extract_radiomics_spine_features(csv_name=image_name, path_to_save=path_to_save)

    def process_all_patients(self):
        # Loop through all patients folders
        for myel_folder in os.listdir(self.base_folder_path):
            myel_path = os.path.join(self.base_folder_path, myel_folder)

            print(f"\nProcessing {myel_folder}")

            # Separate images and masks
            image_files_path, mask_file_path = self._extract_mask(myel_path)

            # Process dataset
            self._create_dataset(image_files_path, mask_file_path, path_to_save=myel_path)


if __name__ == "__main__":
    data_folder_path = r"E:\DATA_Healthy"
    folder_names = "Healthy_"
    ExtractPatientsData(data_folder_path, folder_names).process_all_patients()
