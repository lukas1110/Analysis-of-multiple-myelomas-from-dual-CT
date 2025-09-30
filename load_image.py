import nibabel as nib
import numpy as np


class NiftiFile:
    """
    A utility class to load and work with NIfTI (.nii or .nii.gz) files.
    Provides access to voxel data, header information and real-world coordinates in millimeters.
    """

    def __init__(self, path: str):
        """
        Initialize the NiftiFile instance by loading a NIfTI file.

        Param: path (str): Path to the NIfTI file.
        """
        self.path = path
        self.img = nib.load(path)   # NIfTI object

    def get_image_data(self) -> np.ndarray:
        """
        Return voxel intensities as a NumPy array.

        Returns: numpy.ndarray: Image voxel data.
        """
        return self.img.get_fdata()   # voxel data as numpy array

    def get_image_spacing(self) -> tuple:
        """
        Return voxel spacing (in mm) along each axis.

        Returns: tuple: (x, y, z) in millimeters.
        """
        return tuple(self.img.header['pixdim'][1:4])
