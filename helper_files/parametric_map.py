import os
import SimpleITK as Sitk


base_folder_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diploma thesis"

def extract_images(myel_path):

    image_files_path = []

    # Loop through each file inside the patient folder (Myel_xxx)
    for file_name in os.listdir(myel_path):
        file_path = os.path.join(myel_path, file_name)

        # Skip if it's not a file or not a .nii.gz file
        if not (os.path.isfile(file_path) and file_name.lower().endswith(".nii.gz")):
            continue

        # Separate images and segmentation masks based on filename
        if "seg" in file_name.lower():
            pass
        else:
            image_files_path.append(file_path)

    return image_files_path


for myel_folder in os.listdir(base_folder_path):
    myel_path = os.path.join(base_folder_path, myel_folder)

    # Skip non-folders or folders not starting with "Myel"
    if not os.path.isdir(myel_path) or "myel_" not in myel_folder.lower():
        continue

    image_files_path = extract_images(myel_path)

    for image_path in image_files_path:
        if "monoe_40kev" in image_path:
            vmi_40_path = image_path
        elif "monoe_120kev" in image_path:
            vmi_120_path = image_path
        elif "CaSupp_25" in image_path:
            c_25_path = image_path

    vmi_40 = Sitk.GetArrayFromImage(Sitk.ReadImage(vmi_40_path))
    vmi_120 = Sitk.GetArrayFromImage(Sitk.ReadImage(vmi_120_path))
    c_25 = Sitk.GetArrayFromImage(Sitk.ReadImage(c_25_path))

    img_array = ((vmi_120 + vmi_40) - c_25)
    img = Sitk.GetImageFromArray(img_array)
    img.CopyInformation(Sitk.ReadImage(vmi_40_path))
    Sitk.WriteImage(img, "contrast_image.nii.gz")