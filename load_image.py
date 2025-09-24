import nibabel as nib


def load_nifti_image(path):
    """
    Loads a NIfTI (.nii or .nii.gz) file and returns the image data as a NumPy array.

    Param: path (str): Path to the NIfTI file.

    Returns: data (numpy.ndarray): Image data from the file.
    """
    # Load the NIfTI file
    img = nib.load(path)

    # Convert image object to NumPy array
    data = img.get_fdata()

    return data
