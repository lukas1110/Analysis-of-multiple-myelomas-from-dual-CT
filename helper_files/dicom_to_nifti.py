import os
import SimpleITK as sitk


def dicom_to_nifti(dicom_folder, output_folder):
    myel_name = os.path.basename(output_folder)
    suffix = "_monoe_40kev"
    name = myel_name + suffix

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    nifti_path = os.path.join(output_folder, f"{name}.nii.gz")
    sitk.WriteImage(image, nifti_path)

    print(f"Converted and save to: {nifti_path}")


# dicom_folder = r"E:\DATA_FollowUp_Export\S90320\S204890"
# output_folder = r"E:\DATA_FollowUp\Myel_FollowUp_006_5"
#
# dicom_to_nifti(dicom_folder, output_folder)



# def print_folder_tree(folder, indent=""):
#     items = sorted(os.listdir(folder))
#
#     for i, item in enumerate(items):
#         path = os.path.join(folder, item)
#
#         is_last = i == len(items) - 1
#         branch = "└── " if is_last else "├── "
#
#         print(indent + branch + item)
#
#         if os.path.isdir(path):
#             new_indent = indent + ("    " if is_last else "│   ")
#             print_folder_tree(path, new_indent)
#
#
# main_folder = r"E:\DATA_FollowUp"
#
# print(main_folder)
# print_folder_tree(main_folder)
