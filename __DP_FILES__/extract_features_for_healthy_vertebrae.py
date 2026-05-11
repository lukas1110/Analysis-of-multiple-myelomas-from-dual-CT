import numpy as np
import pandas as pd

import SimpleITK as Sitk
from pathlib import Path

from radiomics import featureextractor


class RadiomicsFeaturesExtraction:
    """
    Handles the extraction of radiomics features from CT images using segmentation masks.
    """
    def __init__(self, ct_img: Sitk.Image, vertebrae_mask: Sitk.Image):
        """
        Initialize the extractor with image data and configuration.

        Args:
            ct_img: The main CT scan as a SimpleITK image.
            vertebrae_mask: The segmentation mask for vertebrae.
        """
        self.ct_img = ct_img
        self.vertebrae_mask_array = Sitk.GetArrayFromImage(vertebrae_mask)

        # Initialize and configure the PyRadiomics feature extractor
        self.features_extractor = self._set_features_extractor(
            featureextractor.RadiomicsFeatureExtractor()
        )

    @staticmethod
    def _set_features_extractor(extractor: featureextractor.RadiomicsFeatureExtractor):
        """
        Configure the radiomics extractor settings and enabled features.
        """
        # Disable unnecessary logging information
        extractor.settings['additionalInfo'] = False
        extractor.disableAllFeatures()

        # Enable specific image types (Original is default, adding Gradient)
        extractor.enableImageTypeByName("Gradient")

        # Enable specific radiomics feature classes
        feature_classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
        for cls in feature_classes:
            extractor.enableFeatureClassByName(cls)

        return extractor

    def _convert_array_to_sitk_image(self, image_data: np.ndarray) -> Sitk.Image:
        """
        Converts a numpy array back to a SimpleITK image, copying geometry from the original CT.
        """
        sitk_image = Sitk.GetImageFromArray(image_data.astype(np.uint8))
        sitk_image.CopyInformation(self.ct_img)
        return sitk_image

    @staticmethod
    def _save_csv(result_list: list, csv_name: str, path_to_save: str = None):
        """
        Exports extracted features to a CSV file.
        """
        df = pd.DataFrame(result_list)
        output_path = Path(path_to_save) / csv_name if path_to_save else Path(csv_name)
        df.to_csv(output_path, index=False)

    def extract_radiomics_spine_features(self, csv_name: str = None, path_to_save: str = None):
        """
        Extracts features for the entire spine based on the provided mask.
        """
        # Create binary mask (everything > 0 is considered spine)
        spine_mask_data = self.vertebrae_mask_array > 0
        spine_sitk_mask = self._convert_array_to_sitk_image(spine_mask_data)

        try:
            # Execute extraction
            features = self.features_extractor.execute(self.ct_img, spine_sitk_mask)
            # Convert OrderedDict to standard dict for easier handling
            result_entry = dict(features)
        except Exception as e:
            print(f"Error extracting spine features for {csv_name}: {e}")
            result_entry = {}

        # Prepare final filename
        filename = f"{csv_name}_radiomics_spine_features.csv" if csv_name else "radiomics_spine_features.csv"

        self._save_csv([result_entry], filename, path_to_save)


class ExtractPatientsData:
    """
    Orchestrates data loading and processing for multiple patient folders.
    """
    def __init__(self, base_folder_path: str, folder_prefix: str):
        """
        Args:
            base_folder_path: Path to the main directory containing patient folders.
            folder_prefix: Prefix to identify relevant patient folders (e.g., "Myel_").
        """
        self.base_folder_path = Path(base_folder_path)
        self.folder_prefix = folder_prefix

    def _extract_image_name(self, path: str) -> str:
        """
        Cleans the image filename by removing extensions and specific prefixes.
        """
        p = Path(path)
        # Handles double extensions like .nii.gz
        image_name = p.name.split('.')[0]

        if image_name.lower().startswith(self.folder_prefix.lower()):
            # Remove "Myel_XX_" prefix based on original logic
            parts = image_name.split("_")
            return "_".join(parts[2:]) if len(parts) > 2 else image_name

        return image_name

    @staticmethod
    def _get_files_from_folder(patient_path: Path):
        """
        Scans a patient folder for image files and the segmentation mask.
        """
        image_paths = []
        mask_path = None

        for file in patient_path.iterdir():
            if not (file.is_file() and file.name.lower().endswith(".nii.gz")):
                continue

            if "seg" in file.name.lower():
                mask_path = file
            else:
                image_paths.append(file)

        return image_paths, mask_path

    def _create_dataset(self, image_paths: list, mask_path: Path, path_to_save: str = None):
        """
        Processes each CT image for a single patient using the shared mask.
        """
        if not mask_path:
            print("No mask found, skipping patient.")
            return

        vertebrae_mask = Sitk.ReadImage(str(mask_path))

        for img_path in image_paths:
            image_name = self._extract_image_name(str(img_path))
            print(f"Current Image: {image_name}")

            # Check if output already exists to avoid redundant computation
            output_folder = Path(path_to_save) if path_to_save else img_path.parent
            expected_csv = output_folder / f"{image_name}_radiomics_spine_features.csv"

            if expected_csv.exists():
                print(f"CSV for {image_name} already exists. Skipping.")
                continue

            # Load and extract
            ct_img = Sitk.ReadImage(str(img_path))
            extractor = RadiomicsFeaturesExtraction(ct_img, vertebrae_mask)

            print("Extracting spine level features...")
            extractor.extract_radiomics_spine_features(csv_name=image_name, path_to_save=path_to_save)

    def process_all_patients(self):
        """
        Main loop iterating through all patient directories in the base path.
        """
        if not self.base_folder_path.exists():
            print(f"Base path {self.base_folder_path} does not exist.")
            return

        for patient_folder in self.base_folder_path.iterdir():
            if not patient_folder.is_dir():
                continue

            print(f"\n{'=' * 20}")
            print(f"Processing Patient Folder: {patient_folder.name}")
            print(f"{'=' * 20}")

            image_paths, mask_path = self._get_files_from_folder(patient_folder)

            # Pass the current patient folder as the save path
            self._create_dataset(image_paths, mask_path, path_to_save=str(patient_folder))


if __name__ == "__main__":
    # Configuration
    DATA_PATH = r"E:\DATA_Healthy_and_Myelom\DATA_MM_Stage3"
    PREFIX = "Myel_"

    # Run Orchestrator
    processor = ExtractPatientsData(DATA_PATH, PREFIX)
    processor.process_all_patients()
