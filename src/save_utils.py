import torch
import os

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"


def get_model_params_dir(experiment_dir, modelname, create_if_nonexistent=False):
    """"
    Get the directory where the model parameters are saved.

    Args:
        experiment_dir:         the directory of the experiment where everything relevant to the experiment is saved.
        modelname:              the name of the model of which we want to get the parameters
        create_if_nonexistent:  boolean that indicates whether we should create the 'directory where the model
                                parameters are saved' if it does not exist yet. DEFAULT: False.

    Returns:
        dir:                    the directory where the model parameters are / should be saved.

    """

    dir = os.path.join(experiment_dir, model_params_subdir, modelname)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):
    """"
    Get the directory where the optimizer parameters are saved.

    Args:
        experiment_dir:         the directory of the experiment where everything relevant to the experiment is saved.
        create_if_nonexistent:  boolean that indicates whether we should create the 'directory where the optimizer
                                parameters are saved' if it does not exist yet. DEFAULT: False.

    Returns:
        dir:                    the directory where the optimizer parameters are / should be saved.

    """

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def save_model(experiment_dir, filename, model, modelname, epoch):
    """"
    Save the parameters of a model and save the epoch at which we save the model.

    Args:
        experiment_dir:         the directory of the experiment where everything relevant to the experiment is saved.
        filename:               the name of the file that saves the parameters of the model and the epoch at which we
                                we save the model
        model:                  the nn.Module module of which we save the parameters
        modelname:              the name of the model
        epoch:                  the epoch at which we save the model parameters

    """

    # Get the directory where to save the parameters
    model_params_dir = get_model_params_dir(experiment_dir, modelname, True)

    # Save the parameters of the model
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_dir, filename, optimizer, epoch):
    """"
    Save the optimizer parameters and save the epoch at which we save these parameters.

    Args:
        experiment_dir:         the directory of the experiment where everything relevant to the experiment is saved.
        filename:               the name of the file that saves the parameters of the model and the epoch at which we
                                we save the model
        optimizer:              the optimizer of which we want to save the parameters
        epoch:                  the epoch at which we save the optimizer parameters

    """

    # Get the directory where to save the parameters
    optimizer_params_dir = get_optimizer_params_dir(experiment_dir, True)

    # Save the optimizer parameters
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_model_parameters(experiment_directory, checkpoint, model, modelname):
    """"
    Load some model by loading some saved parameters.

    Args:
        experiment_dir:         the directory of the experiment where everything relevant to the experiment is saved
        checkpoint:             a specific point (e.g. a certain epoch or the term 'latest') at which parameters were
                                saved
        model:                  the nn.Module model which we want to load the parameters into
        modelname:              the name of the model

    """

    # Get the name of the file which is used to load the parameters
    filename = os.path.join(
        experiment_directory, model_params_subdir, modelname, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    # Load the parameters
    data = torch.load(filename)

    # Put the parameters into the model
    model.load_state_dict(data["model_state_dict"])
