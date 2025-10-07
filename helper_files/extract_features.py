from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as scipy_entropy
import numpy as np


def extract_intensity_features(image_array):
    """
    Compute first-order intensity (histogram) features.

    Param: 1D array of intensity values (non-empty).

    Returns: dict with intensity features
    """
    mean = float(np.mean(image_array))
    std = float(np.std(image_array))
    median = float(np.median(image_array))
    min_voxel = np.min(image_array)
    max_voxel = np.max(image_array)

    p10 = np.percentile(image_array, 10)
    p25 = np.percentile(image_array, 25)
    p75 = np.percentile(image_array, 75)
    p90 = np.percentile(image_array, 90)
    iqr = abs(float(p75) - float(p25))

    sk = skew(image_array)
    kt = kurtosis(image_array)
    rms = np.sqrt(np.mean(image_array ** 2))
    energy = np.sum(image_array ** 2)
    mad = np.mean(np.abs(image_array - mean))

    hist, _ = np.histogram(image_array, bins=256, density=True)
    probs = hist / np.sum(hist)
    entropy = scipy_entropy(probs + np.finfo(float).eps, base=2)
    uni = np.sum(probs ** 2)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": float(min_voxel),
        "max": float(max_voxel),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(iqr),
        "skewness": sk,
        "kurtosis": float(kt),
        "rms": float(rms),
        "energy": float(energy),
        "mad": float(mad),
        "entropy": float(entropy),
        "uniformity": float(uni)
    }


def extract_glcm_features(region_mask, image_data, levels=32):
    """
    Compute GLCM texture features slice-by-slice for a given mask.
    Takes all axial slices that contain voxels from region_mask.

    Param: region_mask : 3D np.array (binary mask of vertebra or lesions)
    Param: image_data : 3D np.array (original CT intensities)
    Param: levels : int, number of gray levels for quantization

    Returns: dict with averaged GLCM features
    """
    features_per_slice = {prop: [] for prop in
                          ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]}

    # Loop through slices
    for z in range(region_mask.shape[2]):
        mask_slice = region_mask[:, :, z]
        if np.sum(mask_slice) == 0:
            continue  # skip empty slices

        img_slice = image_data[:, :, z]
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
                            distances=[1, 2, 3],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=levels,
                            symmetric=True,
                            normed=True)

        for prop in features_per_slice.keys():
            features_per_slice[prop].append(float(graycoprops(glcm, prop).mean()))

    # Average across slices
    averaged = {f"glcm_{prop}": float(np.mean(vals)) if len(vals) > 0 else 0.0
                for prop, vals in features_per_slice.items()}
    return averaged


def extract_shape_features(region_mask, spacing=(0.97, 0.97, 0.6)):
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
