import os
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage


# def dicom_to_nifti(dicom_folder, output_folder):
#     myel_name = os.path.basename(output_folder)
#     suffix = "_monoe_40kev"
#     name = myel_name + suffix
#
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()
#     nifti_path = os.path.join(output_folder, f"{name}.nii.gz")
#     sitk.WriteImage(image, nifti_path)
#
#     print(f"Converted and save to: {nifti_path}")


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


# def label_segmentation_mask(input_file, output_file):
#     # load
#     nii = nib.load(input_file)
#     mask = nii.get_fdata() > 0
#
#     # label objects
#     labels, N = ndimage.label(mask)
#
#     print("objects:", N)
#
#     # save
#     out = nib.Nifti1Image(labels.astype("int32"), nii.affine, nii.header)
#     nib.save(out, output_file)
#
# input_path = r"E:\DATA_FollowUp\Myel_FollowUp_008_4\Myel_FollowUp_008_4_lesions_seg.nii.gz"
#
# label_segmentation_mask(input_path, input_path)


img = sitk.ReadImage(r"E:\DATA_Healthy_and_Myelom\DATA_Healthy\Healthy_010\healthy_010_konv_muscle02_segment.nrrd")
sitk.WriteImage(img, r"E:\DATA_Healthy_and_Myelom\DATA_Healthy\Healthy_010\healthy_010_konv_muscle02_segment.nii.gz")