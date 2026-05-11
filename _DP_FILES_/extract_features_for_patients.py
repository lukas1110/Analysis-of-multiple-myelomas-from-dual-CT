import os

import SimpleITK as Sitk
from pathlib import Path

from radiomics_features_extractor import RadiomicsFeatures


class AllPatientsData:
    """
    Class to process all patient data in a base folder containing Myel_XXX folders.
    Each Myel_XXX folder contains CT images and segmentation masks (.nii.gz files).

    Responsibilities:
    - Extract images and masks for each patient.
    - Identify vertebrae and lesions masks.
    - Process feature extraction and save results to csv.
    """
    def __init__(self, base_folder_path):
        """
        Initialize the dataset processor with the base folder path.

        param: base_folder_path (str): Path to the folder containing Myel_XXX patient folders.
        """
        self.base_folder_path = base_folder_path

    @staticmethod
    def _extract_image_name(path):
        """
        Extract the core name of an image, removing 'myel_XXX_' prefix and '.nii.gz' suffix.

        param: path (str): Full path to the image file.
        returns: str: Clean image name.
        """
        image_name = Path(path).with_suffix("").with_suffix("").name
        if image_name.lower().startswith("Myel_FollowUp_"):
            # Remove 'myel_XXX_' prefix
            return "_".join(image_name.split("_")[2:])
        return image_name

    @staticmethod
    def _extract_masks(myel_path):
        """
        Separate image and mask files in a Myel_XXX folder.

        param: myel_path (str): Path to the Myel_XXX folder.
        returns: tuple: (image_files_path, mask_files_path)
        """
        image_files_path = []
        mask_files_path = []

        # Loop through each file inside the patient folder (Myel_xxx)
        for file_name in os.listdir(myel_path):
            file_path = os.path.join(myel_path, file_name)

            # Skip if it's not a file or not a .nii.gz file
            if not (os.path.isfile(file_path) and file_name.lower().endswith(".nii.gz")):
                continue

            # Separate images and segmentation masks based on filename
            if "seg" in file_name.lower():
                mask_files_path.append(file_path)
            else:
                image_files_path.append(file_path)

        return image_files_path, mask_files_path

    def _create_dataset(self, image_files_path, mask_files_path, path_to_save: str = None):
        """
        Process a single patient's dataset.

        param: image_files_path (list): List of image file paths.
        param: mask_files_path (list): List of mask file paths.
        optional: path_to_save (str): Path to save CSV file.
        """
        vertebrae_mask_path = "None"
        lesions_mask_path = "None"

        # Identify vertebrae and lesion masks
        for mask_path in mask_files_path:
            name = mask_path.lower()
            if "lesions_seg" in name or "lesion" in name:
                lesions_mask_path = mask_path
            elif "spine_seg" in name or "spine" in name:
                vertebrae_mask_path = mask_path

        # Load individual CT images and create dataset using image and masks
        for image_path in image_files_path:
            image_name = self._extract_image_name(image_path)
            print(f"{image_name}")

            # Load images to Sitk objects
            ct_img = Sitk.ReadImage(image_path)
            vertebrae_img = Sitk.ReadImage(vertebrae_mask_path)
            lesions_img = Sitk.ReadImage(lesions_mask_path)

            # Initialize features extractor
            radiomics_features = RadiomicsFeatures(ct_img, vertebrae_img, lesions_img)

            # Define extraction tasks
            tasks = [
                ("spine_vertebrae", "spine", radiomics_features.extract_radiomics_spine_features),
                ("vertebrae", "vertebrae", radiomics_features.extract_radiomics_vertebrae_features),
                ("individual_lesion", "individual lesions",
                 radiomics_features.extract_radiomics_individual_lesion_features)
            ]

            base_dir = os.path.dirname(image_path)

            for suffix, label, extract_method in tasks:
                expected_csv = os.path.join(base_dir, f"{image_name}_radiomics_{suffix}_features.csv")

                if os.path.exists(expected_csv):
                    print(f"CSV {image_name} for {label} already exists, skipping features extraction.")
                else:
                    print(f"{label.capitalize()} level")
                    extract_method(csv_name=image_name, path_to_save=path_to_save)

    def process_all_patients(self):
        """
        Loop through all Myel_XXX folders in the base folder and process each patient.
        """
        # Loop through all Myel_ folders
        for myel_folder in os.listdir(self.base_folder_path):
            myel_path = os.path.join(self.base_folder_path, myel_folder)

            # Skip non-folders or folders not starting with "Myel"
            if not os.path.isdir(myel_path) or "myel" not in myel_folder.lower():
                continue

            print(f"\nProcessing {myel_folder}")

            # Separate images and masks
            image_files_path, mask_files_path = self._extract_masks(myel_path)

            # Process dataset
            self._create_dataset(image_files_path, mask_files_path, path_to_save=myel_path)


if __name__ == "__main__":
    data_folder_path = r"E:\DATA_FollowUp"
    AllPatientsData(data_folder_path).process_all_patients()
