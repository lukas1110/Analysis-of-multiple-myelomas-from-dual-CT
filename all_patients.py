import os
from pathlib import Path


data_folder_path = r"D:\DATA_Myelomy"

def extract_image_name(path):
    image_name = Path(path).with_suffix("").with_suffix("").name
    if image_name.lower().startswith("myel_"):
        return "_".join(image_name.split("_")[2:])
    return image_name

# Loop through all Myel_ folders
for myel_folder in os.listdir(data_folder_path):
    myel_path = os.path.join(data_folder_path, myel_folder)

    if not os.path.isdir(myel_path) or "myel" not in myel_folder.lower():
        continue  # Skip files, we want Myel folders only

    print(f"\nProcessing {myel_folder}")

    # Separate files into two groups (images / masks)
    image_files_path = []
    mask_files_path = []

    # Loop through each file inside Myel_xxx
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

    # Load vertebrae and lesions mask
    vertebrae_mask_path = "None"
    lesions_mask_path = "None"

    # Loop through all mask file paths
    for mask_path in mask_files_path:
        name = mask_path.lower()

        if "lesions_seg" in name or "lesion" in name:
            lesions_mask_path = mask_path
        elif "spine_seg" in name or "spine" in name:
            vertebrae_mask_path = mask_path

    # Load individual CT images
    for image_path in image_files_path:
        print(f"\t{os.path.basename(extract_image_name(image_path))}")
