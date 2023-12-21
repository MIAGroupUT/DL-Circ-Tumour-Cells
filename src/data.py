import torch
import os
import numpy as np


# Get the directory that contains this file and subsequently get the directory on top of this
file_dir = os.path.join(os.path.dirname(__file__), "..")


class Dataset(torch.utils.data.Dataset):
    """"
    This is the Dataset class for performing the experiments with the cellline-data. Currently, it only supports the
    data provided in the repository.

    Args:
        train_or_val:           the value can be 'train' or 'validation' and indicates whether we use a training dataset
                                or a validation dataset.
        number_of_classes:      this integer determines whether we use the case with 2 classes, 5 classes, or 6 classes.
                                Note that these are the only cases available for the data supplied in the repository.

    """

    def __init__(self, train_or_val='train', number_of_classes=2, transforms=None):

        # Checking which dataset to load
        if number_of_classes == 2:
            filename_data = 'xtrain_2class_prepr.npy'
            filename_labels = 'ytrain_2class_prepr.npy'
        elif number_of_classes == 5:
            filename_data = 'xtrain_5class_prepr.npy'
            filename_labels = 'ytrain_5class_prepr.npy'
        elif number_of_classes == 6:
            filename_data = 'xtrain_6class_prepr.npy'
            filename_labels = 'ytrain_6class_prepr.npy'
        else:
            raise ValueError("The value of 'number_of_classes' is {}, but can only be 2, 5, or 6.".format(
                number_of_classes))

        # Checking whether we want to load the training dataset or the validation dataset
        if train_or_val == 'train':
            used_data_partition = 'train_set'
        elif train_or_val == 'validation':
            used_data_partition = 'val_set'
        else:
            raise ValueError("The value of 'train_or_val' is {}, but can only be 'train' or 'validation".format(
                train_or_val))

        # Save the transformations that are going to be used
        self.transform = transforms

        # Define the path where the data and the labels are stored
        dataset = os.path.join(file_dir, 'data', 'cellline-data', used_data_partition, filename_data)
        labels = os.path.join(file_dir, 'data', 'cellline-data', used_data_partition, filename_labels)

        # Load the data numpy file and the labels numpy file
        self.data = np.load(dataset)
        self.labels = np.load(labels)

        # Put both into a torch tensor and rearrange everything properly if needed
        self.data = torch.permute(torch.from_numpy(self.data), (0, 3, 1, 2)) #[(0, 502), ...]
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]

