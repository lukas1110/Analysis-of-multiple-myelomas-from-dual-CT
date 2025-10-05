import os


data_folder_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis"

# Loop through all Myel_ folders
for myel_folder in os.listdir(data_folder_path):
    myel_path = os.path.join(data_folder_path, myel_folder)

    if not os.path.isdir(myel_path) or "myel" not in myel_folder.lower():
        continue  # Skip files, we want folders only

    print(f"\nProcessing {myel_folder}")

    # Separate subfolders into two groups
    image_folders_path = []
    mask_folders_path = []

    # Loop through each subfolder inside Myel_xxx
    for subfolder_path in os.listdir(myel_path):
        sub_path = os.path.join(myel_path, subfolder_path)

        if not os.path.isdir(sub_path):
            continue

        # Separate images and segmentation masks
        if "labels" in subfolder_path.lower():
            mask_folders_path.append(sub_path)
        else:
            image_folders_path.append(sub_path)

    # Load vertebrae and lesions mask
    vertebrae_mask_path = "None"
    lesions_mask_path = "None"

    for mask_folder_path in mask_folders_path:
        if "lesion" in mask_folder_path.lower():
            for file in os.listdir(mask_folder_path):
                if "lesions_seg" in file.lower():
                    lesions_mask_path = os.path.join(mask_folder_path, file)
        elif "spine" in mask_folder_path.lower():
            for file in os.listdir(mask_folder_path):
                if "spine_seg" in file.lower():
                    vertebrae_mask_path = os.path.join(mask_folder_path, file)

    # Load individual CT images
    for image_folder_path in image_folders_path:
        for file in os.listdir(image_folder_path):
            if file.endswith(".nii.gz"):
                image_path = os.path.join(image_folder_path, file)
                print(f"\t{os.path.basename(image_path)} + {os.path.basename(vertebrae_mask_path)} + {os.path.basename(lesions_mask_path)}")
    print()
