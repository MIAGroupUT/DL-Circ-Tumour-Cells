import torch
import os

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"


def get_model_params_dir(experiment_dir, modelname, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir, modelname)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def save_model(experiment_directory, filename, model, modelname, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, modelname, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_model_parameters(experiment_directory, checkpoint, model, modelname):

    filename = os.path.join(
        experiment_directory, model_params_subdir, modelname, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    model.load_state_dict(data["model_state_dict"])

