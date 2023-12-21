import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import os
import random
import warnings

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


def compute_conf_matrix(y_true_val, y_pred_val):
    """"
    Given a numpy array of predicted labels and a numpy array of true labels, create a confusion matrix.

    Args:
        y_true_val:         the numpy array containing the true (integer) labels
        y_pred_val:         the numpy array containing the predicted (integer) labels

    Returns:
        The confusion matrix C
    """

    C = confusion_matrix(y_true_val, y_pred_val)
    return np.matrix(C)


def make_tsne_plot(latent_vec, color_label, label_dict, result_path=0, rect_coords=0, name="tsne2d_plot"):
    """"
    Create a TSNE-plot of the learned latent space.

    Args:
        latent_vec:         a 'batch_size x latent_dim' numpy array where each row contains a latent vector. The
                            latent_vec tensor consists of the latent codes which one wants to make a TSNE-plot.
        color_label:        a 1d numpy array containing the true classes / labels to which each latent vector in
                            latent_vec belongs. Note: the classes / labels are represented by integers.
        label_dict:         a dictionary where each integer in color_label is associated a specific class name. E.g. the
                            label associated with integer 0 could have the name 'WBC'.
        result_path:        if you want to save the TSNE-plot as a .png file, this variable indicates where it is saved.
                            DEFAULT: 0 (which means that by default we do not save the TSNE-plot as a .png file).
        rect_coords:        a list of 4 numbers: [x_left, x_right, y_bottom, y_top]. These numbers form the corners
                            of a rectangle that is plotted inside the TSNE-plot. More precisely, we plot the following
                            rectangle:

                            (x_left, y_top)    ---------------- (x_right, y_top)
                                   |                                    |
                                   |                                    |
                                   |                                    |
                            (x_left, y_bottom) ---------------- (x_right, y_bottom)

                            DEFAULT: 0 (which means that by default we do not plot a rectangle in the TSNE-plot)
        name:               the name of the .png file if we save the TSNE-plot. DEFAULT: tsne2d_plot.

    Returns:
        z_tsne:             the TSNE-plot locations of the latent vectors in latent_vec.

    """

    # Only create the TSNE-plot when the latent dimension is bigger than 2. Else just use the latent vectors and plot
    # those.
    if latent_vec.shape[1] > 2:
        tsne = TSNE(n_components=2, init='pca')
        z_tsne = tsne.fit_transform(latent_vec)
    else:
        z_tsne = latent_vec

    # Get the number of unique labels
    num_unq_labels = len(list(label_dict.keys()))

    # As we only have datasets with 2 classes, 5 classes, and 6 classes, create a colormap
    # for the scatter plot / TSNE-plot. The code below creates a DISCRETE colormap with a color for each class.
    if num_unq_labels == 2:
        cmap = colors.ListedColormap(['red', 'red', 'green', 'green'])
        boundaries = [-0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    elif num_unq_labels == 5:
        cmap = colors.ListedColormap(['red', 'green', 'magenta', 'blue', 'cyan'])
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    elif num_unq_labels == 6:
        cmap = colors.ListedColormap(['red', 'yellow', 'green', 'magenta', 'blue', 'cyan'])
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    else:
        raise ValueError("The current implementation of this repository only allows the provided cellline-data. The "
                         "provided datasets either have 2 classes, 5 classes, or 6 classes. The provided 'label_dict' "
                         "variable has {} classes. So EITHER a new dataset is provided with 0, 1, 3, 4, 7, or more "
                         "classes (which is currently not possible) OR the 'label_dict' input is wrong."
                         "".format(num_unq_labels))

    # Also check whether the amount of predicted classes equals the number of classes in the dataset. If not, return
    # a warning
    if not (num_unq_labels == np.unique(color_label).size):
        warnings.warn("The number of unique predicted labels by the classifier is not equal to the number of "
                      "classes/labels present in the dataset. If you do expect that each class is predicted at least "
                      "once, the used model might not be trained adequately.")

    # If rect_coords is specified, make a rectangle object that we also plot in the TSNE-plot. 
    if rect_coords:
        rect = patches.Rectangle((rect_coords[0], rect_coords[2]), rect_coords[1] - rect_coords[0],
                                 rect_coords[3] - rect_coords[2], linewidth=2, edgecolor='black', facecolor='none')

    # Create the figure object
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    # Create the scatter plot with the correct colormap
    sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=color_label, s=10, cmap=cmap, norm=norm)

    # Create a colorbar next to the scatter plot needed to indicate what label corresponds to which color
    cb = plt.colorbar(sc)
    cb.set_ticks(np.arange(0, num_unq_labels, 1), labels=[label_dict[i] for i in list(label_dict.keys())])

    # Add the rectangle.
    if rect_coords:
        ax.add_patch(rect)

    # Save the figure if the result_path variable is supplied
    if result_path:
        plt.savefig(os.path.join(result_path, name + ".png"))

    # Show the TSNE-plot and return z_tsne (which contains the locations of the latent vectors in the TSNE-plot).
    plt.show()
    return z_tsne


def trace_tsne_cluster(tsne_code, color_label, label_dict, x, tsne1low, tsne1up, tsne2low, tsne2up, nrex, result_path=0,
                       name="reconstruction_of_tsne_selection"):
    """"
    This function grabs an earlier calculated TSNE-plot, plots a rectangular box somewhere in this plot, and grabs some
    ground truth images corresponding to TSNE-plot points inside this box. Subsequently, it plots these ground truth
    images.

    Args:
        tsne_code:          a 'batch_size x 2' numpy array containing the points in the TSNE-plot. These correspond to
                            the locations of the latent vectors in the TSNE-plot. The number 'batch_size' indicates the
                            number of points in the TSNE-plot.
        color_label:        a 1d numpy array containing the true classes / labels to which each latent vector in
                            latent_vec belongs. Note: the classes / labels are represented by integers.
        label_dict:         a dictionary where each integer in color_label is associated a specific class name. E.g. the
                            label associated with integer 0 could have the name 'WBC'
        x:                  a numpy array of size 'batch_size x image_width x image_height'. It contains the (ground
                            truth) data / images corresponding to each point in the TSNE-plot.
        tsne1low:           the x-value of the left-side of the rectangular box. Corresponds to x_left in the
                            rect_coords input of 'make_tsne_plot'.
        tsne1up:            the x-value of the right-side of the rectangular box. Corresponds to x_right in the
                            rect_coords input of 'make_tsne_plot'.
        tsne2low:           the y-value of the bottom-side of the rectangular box. Corresponds to y_bottom in the
                            rect_coords input of 'make_tsne_plot'.
        tsne2up:            the y-value of the top-side of the rectangular box. Corresponds to y_top in the
                            rect_coords input of 'make_tsne_plot'.
        nrex:               the number of randomly sampled points in the rectangular box OF which we want to show the
                            corresponding (ground truth) image.
        result_path:        if you want to save the plot with the (ground truth) images as a .png file, this variable
                            indicates where it is saved. DEFAULT: 0 (which means that by default we do not save the
                            plot as a .png file).
        name:               the name of the .png file if we save the plot of the (ground truth images) that correspond
                            to points inside the rectangular box. DEFAULT: reconstruction_of_tsne_selection.
    """

    # Highlight the selected area in the TSNE-plot / scatter plot
    _ = make_tsne_plot(tsne_code, color_label, label_dict, rect_coords=[tsne1low, tsne1up, tsne2low, tsne2up])

    # Find the points in the TSNE-plot that are inside the box
    tsne1_fulfilled = np.logical_and(tsne_code[:, 0] >= tsne1low, tsne_code[:, 0] <= tsne1up)
    tsne2_fulfilled = np.logical_and(tsne_code[:, 1] >= tsne2low, tsne_code[:, 1] <= tsne2up)
    in_gate = np.logical_and(tsne1_fulfilled, tsne2_fulfilled)

    # Grab the points in the TSNE-plot that are inside the box
    selection = x[in_gate,]

    # There are selection.shape[0] points inside the (rectangular) box. Get the number of points that we want to sample
    # within this box. More precisely, we sample 'nrex' amount of points in case nrex <= selection.shape[0]. If
    # nrex > selection.shape[0], we grab all points inside the box.
    num = int(min(selection.shape[0], nrex))

    # Get 'num' random indices. These random indices will be used to grab 'num' random points that are inside the box.
    to_vis = random.sample(range(0, selection.shape[0]), min(selection.shape[0], nrex))

    # Create the figure object
    plt.figure(figsize=(10, np.ceil(nrex / 2) * 5))

    # For every object that we sample from the rectangular box, we do ...
    for i in range(num):

        # Activate the correct subplot
        plt.subplot(num, 2, i + 1)

        # Grab the image corresponding to the i-th sampled data point from the box
        x_input = np.squeeze(selection[to_vis[i]] / np.max(selection[to_vis[i]]) * 255)
        x_input = x_input.astype('uint8')
        x_input = np.dstack((x_input[:, :, 0], x_input[:, :, 2], x_input[:, :, 1]))

        # Show the image
        plt.imshow(x_input)

    # Save the just created plot
    if result_path:
        plt.savefig(os.path.join(result_path, name + ".png"))

    # Show the just created plot
    plt.show()


def save_reconstruction_and_gt_images(save_dir: str, reconstructions: np.ndarray, ground_truths: np.ndarray):
    """"
    This is a function that takes a batch of (reconstructed image, ground truth image) pairs, plots the images
    side-by-side, and saves the corresponding figures in 'save_dir'. It saves the i-th
    (reconstructed image, ground truth image) pair in 'save_dir' with the name 'Reconstruction_vs_GT_pair_i'.

    Args:
        save_dir:           the directory where we save the images
        reconstructions:    a numpy array of shape 'batch_size x width x height x 3' containing a batch of
                            reconstructed images.
        ground_truths:      a numpy array of shape 'batch_size x width x height x 3' containing a batch of
                            ground truth images.
                            The i-th image-pair we plot is (reconstructions[i-1, ...], ground_truths[i-1, ...]).

    """

    # Check that the last dimension corresponds to the RGB channel. Also check whether the tensors are 4D tensors (as
    # they should be).
    if not (reconstructions.shape[-1] == 3 and ground_truths.shape[-1] == 3):
        raise ValueError("The last dimension of the reconstructed images and the ground truth images should be 3. "
                         "Now the shape of the reconstructed images tensor is {} and the shape of the ground truth"
                         "images tensor is {}. Possibly rearrange the dimensions such that the last dimension "
                         "corresponds to the RGB channel!".format(reconstructions.shape, ground_truths.shape))
    elif not (len(reconstructions.shape) == 4 and len(ground_truths.shape) == 4):
        raise ValueError("The shape of the reconstructed images tensor is {} and the shape of the ground truth images "
                         "tensor is {}. But both should be 4D tensors with dimensions batch_size x width x height x 3."
                         .format(reconstructions.shape, ground_truths.shape))

    # Create the save directory if it does not exist yet
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # For every (reconstruction, gt) image pair, do ...
    for i in range(reconstructions.shape[0]):

        # Grab the first reconstructed image and the corresponding ground truth
        reconstruction = reconstructions[i, ...]
        gt = ground_truths[i, ...]

        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Put the images in there
        ax1.imshow(reconstruction)
        ax2.imshow(gt)

        # Add some titles
        ax1.title.set_text("Reconstruction")
        ax2.title.set_text("Ground truth")

        # Save the figure
        plt.savefig(os.path.join(save_dir, "Reconstruction_vs_GT_pair_{}".format(i+1)))
