from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as scipy_entropy
import numpy as np


def compute_intensity_features(data_array):
    """
    Compute first-order intensity (histogram) features.

    Param: 1D array of intensity values (non-empty).

    Returns: dict with intensity features
    """

    mean = np.mean(data_array)
    std = np.std(data_array)
    median = np.median(data_array)
    min = np.min(data_array)
    max = np.max(data_array)


    p10 = np.percentile(data_array, 10)
    p25 = np.percentile(data_array, 25)
    p75 = np.percentile(data_array, 75)
    p90 = np.percentile(data_array, 90)
    iqr = abs(p75 - p25)

    sk = skew(data_array)
    kt = kurtosis(data_array)
    rms = np.sqrt(np.mean(data_array ** 2))
    energy = np.sum(data_array ** 2)
    mad = np.mean(np.abs(data_array - mean))

    hist, _ = np.histogram(data_array, bins=256, density=True)
    probs = hist / np.sum(hist)
    H = scipy_entropy(probs + np.finfo(float).eps, base=2)
    uni = np.sum(probs ** 2)

    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "min": float(min),
        "max": float(max),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(iqr),
        "skewness": float(sk),
        "kurtosis": float(kt),
        "rms": float(rms),
        "energy": float(energy),
        "mad": float(mad),
        "entropy": float(H),
        "uniformity": float(uni)
    }


def compute_glcm_features_slicewise(region_mask, img_data, levels=32,
                                    distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
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


def compute_shape_features(region_mask, spacing=(0.9, 0.9, 0.9)):
    """
    Compute simple 3D shape features from binary mask.

    Param: region_mask : 3D np.array (binary mask)
    Param: spacing : tuple, voxel spacing in (x,y,z) [mm]

    Returns: dict with shape features
    """
    voxel_volume = np.prod(spacing)
    volume = np.sum(region_mask) * voxel_volume

    coords = np.argwhere(region_mask)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    bbox_dims = (max_coords - min_coords + 1) * spacing

    return {
        "volume_mm3": float(volume),
        "bbox_x_mm": float(bbox_dims[0]),
        "bbox_y_mm": float(bbox_dims[1]),
        "bbox_z_mm": float(bbox_dims[2])
    }
