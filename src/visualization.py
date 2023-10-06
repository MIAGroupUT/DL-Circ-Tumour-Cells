import matplotlib.pyplot as plt
import numpy as np
import os


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

