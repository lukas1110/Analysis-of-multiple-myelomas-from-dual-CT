import SimpleITK as sitk
import numpy as np


def load_nifti(path):
    image = sitk.ReadImage(path)
    return image

def resample_image(image, target_spacing=(0.7, 0.7, 0.9), interpolator=sitk.sitkNearestNeighbor):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)

def intensity_normalization(image, clip_range=(-1000, 1000)):
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    # clip HU
    array = np.clip(array, clip_range[0], clip_range[1])

    # # z-score
    # mean = np.mean(array)
    # std = np.std(array) + 1e-8
    # array = (array - mean) / std

    new_image = sitk.GetImageFromArray(array)
    new_image.CopyInformation(image)

    return new_image

def discretization(image, bin_width=25):
    array = sitk.GetArrayFromImage(image)

    discretized = np.floor((array - array.min()) / bin_width)

    new_image = sitk.GetImageFromArray(discretized)
    new_image.CopyInformation(image)

    return new_image

def save_nifti(image, path):
    sitk.WriteImage(image, path)


if __name__ == "__main__":
    input_image_path = r"E:\DATA_Healthy_and_Myelom\DATA_MM_Stage1\Myel_001\myel_001_konv.nii.gz"
    output_image_path = r"E:\DATA_Healthy_and_Myelom\Resampled\myel_001_konv_resampled_02.nii.gz"

    input_mask_path = r"E:\DATA_Healthy_and_Myelom\DATA_MM_Stage1\Myel_001\myel_001_spine_seg.nii.gz"
    output_mask_path = r"E:\DATA_Healthy_and_Myelom\Resampled\myel_001_spine_seg_resampled_02.nii.gz"

    img = load_nifti(input_image_path)
    mask = load_nifti(input_mask_path)

    # 1. resampling
    img_resampled = resample_image(img)
    mask_resampled = resample_image(mask)

    # # 2. intensity normalization
    # img_normalized = intensity_normalization(img_resampled)
    #
    # # 3. discretization
    # img_discretized = discretization(img_normalized)

    # save
    save_nifti(img_resampled, output_image_path)
    save_nifti(mask_resampled, output_mask_path)

    print("Done.")