import os
import numpy as np
import torch
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from src.models import Encoder, Decoder, Classifier
from src.save_utils import load_model_parameters
from evaluation_utils import make_tsne_plot, compute_conf_matrix, trace_tsne_cluster

data_path = os.path.join("..", "data", "cellline-data", "val_set")


def load_data():

    # Load the different validation datasets depending on the number of classes used
    x_val_2class = np.load(os.path.join(data_path, "xval_2class_prepr.npy"))
    y_val_2class = np.load(os.path.join(data_path, "yval_2class_prepr.npy"))
    label_dict_2class = {0: "noCTC", 1: "CTC"}

    x_val_5class = np.load(os.path.join(data_path, "xval_5class_prepr.npy"))
    y_val_5class = np.load(os.path.join(data_path, "yval_5class_prepr.npy"))
    label_dict_5class = {0: "WBC", 1: "CTC", 2: "artifact", 3: "nucl", 4: "tdEV"}

    x_val_6class = np.load(os.path.join(data_path, "xval_6class_prepr.npy"))
    y_val_6class = np.load(os.path.join(data_path, "yval_6class_prepr.npy"))
    label_dict_6class = {0: "WBC", 1: "CD45EV", 2: "CTC", 3: "artifact", 4: "nucl", 5: "tdEV"}

    # Return all this data
    return x_val_2class, y_val_2class, label_dict_2class, \
           x_val_5class, y_val_5class, label_dict_5class, \
           x_val_6class, y_val_6class, label_dict_6class


def plot_and_save_confusion_matrix(conf_mat, labels, result_path=0):

    # Plot the confusion matrix
    df_cm = pd.DataFrame(conf_mat, index=[labels[i] for i in range(len(labels))],
                         columns=[labels[i] for i in range(len(labels))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    # Save it in case a result_path is given
    if result_path:
        plt.savefig(os.path.join(result_path, "confusion_matrix.png"))


def evaluate_model_on_data(model_path, latent_dim, number_of_classes):

    # Use cuda if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load the data
    x_val_2class, y_val_2class, label_dict_2class, \
    x_val_5class, y_val_5class, label_dict_5class, \
    x_val_6class, y_val_6class, label_dict_6class = load_data()

    # Create the different models
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    classifier = Classifier(latent_dim, number_of_classes).to(device)

    # Load the model parameters
    load_model_parameters(model_path, "latest", encoder, "encoder")
    load_model_parameters(model_path, "latest", decoder, "decoder")
    load_model_parameters(model_path, "latest", classifier, "classifier")

    # Create a directory where we will save the created figures
    result_path = os.path.join(model_path, "evaluation_figures")
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # Depending on the number of classes we trained our model on, we do a different kind of evaluation / analysis.
    if number_of_classes == 2:

        # In case of 2 classes we create a t-sne plot with colors according to the true 2-class labels and the true
        # 6-class labels

        # Compute latent representation on the 2 class data
        input = torch.from_numpy(x_val_2class).to(device).permute((0, 3, 1, 2))
        lat_codes = encoder(input).detach().cpu().numpy()

        # Make a tsne plot with the 2 class labels
        tsne_x_val_2class = make_tsne_plot(lat_codes, y_val_2class, label_dict_2class, result_path=result_path,
                                           name="tsne_2class_with_2class_true_label_coloring")

        # Make the tsne plot with the 6 class labels
        _ = make_tsne_plot(tsne_x_val_2class, y_val_6class, label_dict_6class, result_path=result_path,
                           name="tsne_2class_with_6class_true_label_coloring")

    elif number_of_classes == 5:

        # Use the encoder to encode and predict the 6 class data
        input = torch.from_numpy(x_val_6class).to(device).permute((0, 3, 1, 2))
        lat_codes_6class = encoder(input)
        probs = classifier(lat_codes_6class)
        y_pred_val = np.argmax(probs.detach().cpu().numpy(), axis=1)

        # Create a tsne plot with the predicted labels
        tsne_x_val = make_tsne_plot(lat_codes_6class.detach().cpu().numpy(), y_pred_val, label_dict_5class,
                                    result_path=result_path, name="tsne_5class_with_predicted_label_coloring")

        # Reconstruct 10 examples of cluster -3 <= tsne1 <= 7 & 2 <= tsne2 <= 13
        trace_tsne_cluster(tsne_x_val, x_val_6class, -3, 7, 2, 13, 10, result_path=result_path,
                           name="trace_tsne_cluster")

    elif number_of_classes == 6:

        # Use the encoder to encode and predict the 6 class data
        input = torch.from_numpy(x_val_6class).to(device).permute((0, 3, 1, 2))
        lat_codes_6class = encoder(input)
        probs = classifier(lat_codes_6class)
        y_pred_val = np.argmax(probs.detach().cpu().numpy(), axis=1)

        # Compute and print confusion matrix of validation set
        C = compute_conf_matrix(y_val_6class, y_pred_val)
        print("Confusion matrix 6 class set:")
        print(C)

        # Also plot the confusion matrix and save the resulting figure
        plot_and_save_confusion_matrix(C, label_dict_6class, result_path=result_path)

        # Create a tsne plot with the true labels
        _ = make_tsne_plot(lat_codes_6class.detach().cpu().numpy(), y_val_6class, label_dict_6class,
                           result_path=result_path, name="tsne_6class_with_6class_true_label_coloring")

    else:
        raise ValueError("the 'number_of_classes' variable is {}, while it can only be 2, 5 or 6."
                         .format(number_of_classes))


if __name__ == "__main__":
    model_path = os.path.join("..", "results", "test", "model_beta_1000")
    latent_dim = 50
    number_of_classes = 5

    evaluate_model_on_data(model_path, latent_dim, number_of_classes)
