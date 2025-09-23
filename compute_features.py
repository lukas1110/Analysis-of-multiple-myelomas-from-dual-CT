from skimage.feature import graycomatrix, graycoprops
import numpy as np


def compute_glcm_features_slicewise(region_mask, img_data, levels=32, distances=[1], angles=[0]):
    """
    Compute GLCM texture features slice-by-slice for a given mask.
    Takes all axial slices that contain voxels from region_mask.

    Param: region_mask : 3D np.array (binary mask of vertebra or lesions)
    Param: img_data : 3D np.array (original CT intensities)
    Param: levels : int, number of gray levels for quantization
    Param: distances : list, pixel distances for GLCM
    Param: angles : list, angles for GLCM

    Returns: dict with averaged GLCM features
    """
    features_per_slice = {prop: [] for prop in
                          ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]}

    # Loop through slices
    for z in range(region_mask.shape[2]):
        mask_slice = region_mask[:, :, z]
        if np.sum(mask_slice) == 0:
            continue  # skip empty slices

        img_slice = img_data[:, :, z]
        voxels = img_slice[mask_slice > 0]

        # quantize intensities into [0, levels-1]
        r_min, r_max = voxels.min(), voxels.max()
        if r_max == r_min:
            continue  # skip uniform slices
        scaled = np.floor((voxels - r_min) / (r_max - r_min) * (levels - 1)).astype(np.uint8)

        # create small 2D image (fill only mask area, background=0)
        img_quantized = np.zeros_like(mask_slice, dtype=np.uint8)
        img_quantized[mask_slice > 0] = scaled

        # compute GLCM for this slice
        glcm = graycomatrix(img_quantized,
                            distances=distances,
                            angles=angles,
                            levels=levels,
                            symmetric=True,
                            normed=True)

        for prop in features_per_slice.keys():
            features_per_slice[prop].append(float(graycoprops(glcm, prop).mean()))

    # Average across slices
    averaged = {f"glcm_{prop}": float(np.mean(vals)) if len(vals) > 0 else 0.0
                for prop, vals in features_per_slice.items()}
    return averaged
