import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def show_individual_slices(data):
    """
    Show the middle axial, sagittal, and coronal slices
    from a 3D NIfTI (NumPy array).

    Param: data (numpy.ndarray): 3D image array (X, Y, Z).
    """
    plt.figure()

    # Axial slice (XY plane at middle Z)
    plt.subplot(131)
    plt.imshow(data[:, :, data.shape[2] // 2].T, cmap="gray", origin="lower")
    plt.title("Axial view")
    plt.axis("off")

    # Sagittal slice (YZ plane at middle X)
    plt.subplot(132)
    plt.imshow(data[data.shape[0] // 2, :, :].T, cmap="gray", origin="lower")
    plt.title("Sagittal view")
    plt.axis("off")

    # Coronal slice (XZ plane at middle Y)
    plt.subplot(133)
    plt.imshow(data[:, data.shape[1] // 2, :].T, cmap="gray", origin="lower")
    plt.title("Coronal view")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_slices_scroll(data, axis=0, pause_time=0.01):
    """
    Scroll through all slices of a 3D NIfTI (NumPy array) along
    the chosen axis with provided time step.

    Param: data (numpy.ndarray): 3D image array (X, Y, Z).
    Param: axis (int): Axis to scroll along:
           0 = sagittal, 1 = coronal, 2 = axial (default = 0).
    Param: pause_time (float): Time step between slices in seconds (default = 0.01).
    """
    plt.figure()

    for i in range(data.shape[axis]):
        if axis == 0:      # sagittal
            slice_2d = data[i, :, :]
        elif axis == 1:    # coronal
            slice_2d = data[:, i, :]
        else:              # axial
            slice_2d = data[:, :, i]

        # Plot the slice
        plt.imshow(slice_2d.T, cmap="gray", origin="lower")
        plt.title(f"Slice {i+1}/{data.shape[axis]}")
        plt.axis("off")
        plt.pause(pause_time)  # wait
        plt.clf()              # clear for next frame

    plt.close()


def show_slices_scroll_with_mask(data, mask_data, axis=0, pause_time=0.025):
    """
    Scroll through slices of a 3D image with an overlaid segmentation mask.

    Param: data (numpy.ndarray): 3D image array (X, Y, Z).
    Param: mask_data (numpy.ndarray): 3D mask array of the same shape (e.g. lesions).
    Param: axis (int): Axis to scroll along:
           0 = sagittal, 1 = coronal, 2 = axial (default = 0).
    Param: pause_time (float): Time step between slices in seconds (default = 0.025).
    """
    # Make sure mask is binary (0 or 1)
    mask_data = (mask_data > 0).astype(np.uint8)

    # Custom colormap for mask: transparent background, red overlay
    cmap = matplotlib.colormaps.get_cmap("autumn").copy()
    cmap.set_under(color="black", alpha=0)   # values <0.5 transparent
    cmap.set_over(color="red", alpha=0.6)    # values >=0.5 red

    plt.figure()

    for i in range(data.shape[axis]):
        if axis == 0:      # sagittal
            slice_img = data[i, :, :]
            slice_mask = mask_data[i, :, :]
        elif axis == 1:    # coronal
            slice_img = data[:, i, :]
            slice_mask = mask_data[:, i, :]
        else:              # axial
            slice_img = data[:, :, i]
            slice_mask = mask_data[:, :, i]

        # Show image slice
        plt.imshow(slice_img.T, cmap="gray", origin="lower")

        # Overlay mask
        plt.imshow(slice_mask.T, cmap=cmap, alpha=0.6,
                   origin="lower", vmin=0.5, vmax=1.5)

        plt.title(f"Slice {i+1}/{data.shape[axis]}")
        plt.axis("off")
        plt.pause(pause_time)
        plt.clf()

    plt.close()


def show_slices_scroll_range_with_mask(data, mask_data, axis=0, pause_time=0.025, start=None, end=None):
    """
    Scroll through a range of slices of a 3D image with an overlaid segmentation mask.

    Param: data (numpy.ndarray): 3D image array (X, Y, Z).
    Param: mask_data (numpy.ndarray): 3D mask array of the same shape (e.g. lesions).
    Param: axis (int): Axis to scroll along:
           0 = sagittal, 1 = coronal, 2 = axial (default = 0).
    Param: pause_time (float): Time step between slices in seconds (default = 0.025).
    Param: start (int): Starting slice index (default = 0).
    Param: end (int): Ending slice index (default = last slice).
    """
    # Make sure mask is binary (0 or 1)
    mask_data = (mask_data > 0).astype(np.uint8)

    # Custom colormap for mask: transparent background, red overlay
    cmap = matplotlib.colormaps.get_cmap("autumn").copy()
    cmap.set_under(color="black", alpha=0)   # values <0.5 transparent
    cmap.set_over(color="red", alpha=0.6)    # values >=0.5 red

    # Default range if not given
    if start is None:
        start = 0
    if end is None or end > data.shape[axis]:
        end = data.shape[axis]

    plt.figure()

    for i in range(start, end):
        if axis == 0:      # sagittal
            slice_img = data[i, :, :]
            slice_mask = mask_data[i, :, :]
        elif axis == 1:    # coronal
            slice_img = data[:, i, :]
            slice_mask = mask_data[:, i, :]
        else:              # axial
            slice_img = data[:, :, i]
            slice_mask = mask_data[:, :, i]

        # Show image slice
        plt.imshow(slice_img.T, cmap="gray", origin="lower")

        # Overlay mask
        plt.imshow(slice_mask.T, cmap=cmap, alpha=0.6,
                   origin="lower", vmin=0.5, vmax=1.5)

        plt.title(f"Slice {i+1}/{end}")
        plt.axis("off")
        plt.pause(pause_time)
        plt.clf()

    plt.close()
