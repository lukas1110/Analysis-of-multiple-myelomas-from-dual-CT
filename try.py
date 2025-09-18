import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import numpy as np


# Path to your .nii.gz file
file_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\ConvCT_data_nifti\myel_001_konv.nii.gz"
mask_path = r"C:\Users\lukas\OneDrive - VUT\Plocha\Diplomová práca\Myel_001\Lesion_labels\Myel_001_lesions_seg_validation_VV_final.nii.gz"

# Load the NIfTI file
img = nib.load(file_path)
mask_img = nib.load(mask_path)

# Get the image data as a NumPy array
data = img.get_fdata()
mask_data = mask_img.get_fdata()

# Make custom colormap: 0 = transparent, 1 = red/orange
mask_data = (mask_data > 0).astype(np.uint8)
cmap = matplotlib.colormaps.get_cmap("autumn").copy()
cmap.set_under(color="black", alpha=0)   # everything <0.5 is transparent
cmap.set_over(color="red", alpha=0.6)    # 1 becomes red overlay

# print("Data shape:", data.shape)
# print("Data type:", data.dtype)

# # You can also access metadata (header info)
# print(img.header)




plt.figure()

# Axial slice
plt.subplot(131)
plt.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
plt.title("Axial view")

# Sagittal slice
plt.subplot(132)
plt.imshow(data[:, data.shape[1] // 2, :].T, cmap="gray", origin="lower")
plt.title("Sagittal view")

# Coronal slice
plt.subplot(133)
plt.imshow(data[data.shape[0] // 2, :, :].T, cmap="gray", origin="lower")
plt.title("Coronal view")

plt.show()





# # Plot the entire view slice by slice
# # Choose which axis to scroll
# axis = 0   # sagittal axis
# # axis = 1   # coronal axis
# # axis = 2   # axial axis
#
# # Create a figure
# plt.figure()
#
# for i in range(data.shape[axis]):
#     if axis == 0:  # sagittal
#         slice_2d = data[i, :, :]
#     elif axis == 1:  # coronal
#         slice_2d = data[:, i, :]
#     else:  # axial
#         slice_2d = data[:, :, i]
#
#     # Plot slice
#     plt.imshow(slice_2d.T, cmap="gray", origin="lower")
#     plt.title(f"Slice {i+1}/{data.shape[axis]}")
#     plt.axis("off")
#     plt.pause(0.01)  # time step in seconds
#     plt.clf()        # clear for next frame
#
# plt.close()





# # Plot slice by slice with the binary mask
# axis = 0
# plt.figure()
#
# for i in range(data.shape[axis]):
#     if axis == 0:  # sagittal
#         slice_img = data[i, :, :]
#         slice_mask = mask_data[i, :, :]
#     elif axis == 1:  # coronal
#         slice_img = data[:, i, :]
#         slice_mask = mask_data[:, i, :]
#     else:  # axial
#         slice_img = data[:, :, i]
#         slice_mask = mask_data[:, :, i]
#
#     # Show original slice
#     plt.imshow(slice_img.T, cmap="gray", origin="lower")
#
#     # Overlay vertebrae (mask==1)
#     plt.imshow(slice_mask.T, cmap=cmap, alpha=0.6, origin="lower", vmin=0.5, vmax=1.5)
#
#     plt.title(f"Slice {i + 1}/{data.shape[axis]}")
#     plt.axis("off")
#     plt.pause(0.025)
#     plt.clf()
#
# plt.close()