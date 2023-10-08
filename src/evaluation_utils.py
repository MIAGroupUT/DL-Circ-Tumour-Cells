import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import os
import random

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


def compute_conf_matrix(y_true_val, y_pred_val):
    C = confusion_matrix(y_true_val, y_pred_val)
    return np.matrix(C)


def make_tsne_plot(latent_vec, color_label, label_dict=[], result_path=0, rect_coords=0, name="tsne2d_plot"):
    if latent_vec.shape[1] > 2:
        tsne = TSNE(n_components=2, init='pca')
        z_tsne = tsne.fit_transform(latent_vec)
    else:
        z_tsne = latent_vec
    unq = np.unique(color_label)

    if unq.size == 2:
        cmap = colors.ListedColormap(['red', 'red', 'green', 'green'])
        boundaries = [-0.5, 0.5, 1.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    elif unq.size == 5:
        cmap = colors.ListedColormap(['red', 'green', 'magenta', 'blue', 'cyan'])
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    elif unq.size == 6:
        cmap = colors.ListedColormap(['red', 'yellow', 'green', 'magenta', 'blue', 'cyan'])
        boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    else:
        cmap = plt.cm.get_cmap('jet')
        norm = None

    if rect_coords:
        rect = patches.Rectangle((rect_coords[0], rect_coords[2]), rect_coords[1] - rect_coords[0],
                                 rect_coords[3] - rect_coords[2], linewidth=2, edgecolor='black', facecolor='none')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    if norm is None:
        sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=color_label, vmin=0, vmax=unq.size - 1, s=10, cmap=cmap)
    else:
        sc = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=color_label, s=10, cmap=cmap, norm=norm)
    cb = plt.colorbar(sc)
    cb.set_ticks(np.arange(0, unq.size, 1))
    cb.set_ticklabels(label_dict)
    if rect_coords:
        ax.add_patch(rect)
    if result_path:
        plt.savefig(os.path.join(result_path, name + ".png"))
    plt.show()
    return z_tsne


def trace_tsne_cluster(tsne_code, x, tsne1low, tsne1up, tsne2low, tsne2up, nrex, result_path=0,
                       name="reconstruction_of_tsne_selection"):
    # highlight selected area in scatter plot
    _ = make_tsne_plot(tsne_code, np.ones(tsne_code.shape[0]), rect_coords=[tsne1low, tsne1up, tsne2low, tsne2up])

    tsne1_fulfilled = np.logical_and(tsne_code[:, 0] >= tsne1low, tsne_code[:, 0] <= tsne1up)
    tsne2_fulfilled = np.logical_and(tsne_code[:, 1] >= tsne2low, tsne_code[:, 1] <= tsne2up)
    in_gate = np.logical_and(tsne1_fulfilled, tsne2_fulfilled)

    selection = x[in_gate,]
    to_vis = random.sample(range(0, selection.shape[0]), min(selection.shape[0], nrex))

    # display pairs of intput/output
    num = int(min(selection.shape[0], nrex))

    plt.figure(figsize=(10, np.ceil(nrex / 2) * 5))
    for i in range(num):
        plt.subplot(num, 2, i + 1)
        x_input = np.squeeze(selection[to_vis[i]] / np.max(selection[to_vis[i]]) * 255)
        x_input = x_input.astype('uint8')
        x_input = np.dstack((x_input[:, :, 0], x_input[:, :, 2], x_input[:, :, 1]))
        plt.imshow(x_input)
    if result_path:
        plt.savefig(os.path.join(result_path, name + ".png"))
    plt.show()

    return