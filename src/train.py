from src.models import Encoder, Decoder, Classifier
from src.data import Dataset
from src.save_utils import save_model, save_optimizer, load_model_parameters
from src.visualization import save_reconstruction_and_gt_images

from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomAffine, InterpolationMode
from torch.utils.data.dataloader import DataLoader

import torch

import os
import json

import wandb

cross_entropy_loss = torch.nn.CrossEntropyLoss()

binary_cross_entropy_loss = torch.nn.BCELoss(reduction='mean')

negative_log_likelihood = torch.nn.NLLLoss(reduction='mean')


def train(experiment_directory, prev_model_dir=None, latent_dim=50, num_epochs=75, batch_size=500, number_of_classes=2,
          beta=1.0, alpha=0.001, gamma=1.0, log_frequency=100, snapshot_frequency=100, num_random_samples=3,
          batch_size_update_freq=125, max_batch_size=256):

    # Use cuda if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # First, create the different models
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    classifier = Classifier(latent_dim, number_of_classes).to(device)

    # Load the model parameters of a previously trained model if you want to
    if prev_model_dir is not None:
        load_model_parameters(prev_model_dir, "latest", encoder, "encoder")
        load_model_parameters(prev_model_dir, "latest", decoder, "decoder")
        load_model_parameters(prev_model_dir, "latest", classifier, "classifier")

    # Define the transformations used for data augmentation
    transforms = Compose([
        RandomHorizontalFlip(0.5),
        RandomAffine(40, translate=(0.15, 0.15), shear=0.05, scale=(1-0.2, 1+0.2), interpolation=InterpolationMode.NEAREST)
    ])

    #transforms = None

    # Load the dataset
    dataset = Dataset('train', number_of_classes, transforms)

    # Get the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define the optimizers

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) +
                                 list(classifier.parameters()), lr=0.001)

    # NOTE: In the original paper, the optimizer was 'Adadelta'. I used Adam instead!

    # optimizer = torch.optim.Adadelta(
    #     [
    #         {
    #             "params": encoder.parameters(),
    #             "lr": 0.001,
    #         },
    #         {
    #             "params": decoder.parameters(),
    #             "lr": 0.001,
    #         },
    #         {
    #             "params": classifier.parameters(),
    #             "lr": 0.001,
    #         },
    #     ]
    # )

    # Define two saving functions
    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", encoder, "encoder", epoch)
        save_model(experiment_directory, "latest.pth", decoder, "decoder", epoch)
        save_model(experiment_directory, "latest.pth", classifier, "classifier", epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", encoder, "encoder", epoch)
        save_model(experiment_directory, str(epoch) + ".pth", decoder, "decoder", epoch)
        save_model(experiment_directory, str(epoch) + ".pth", classifier, "classifier", epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer, epoch)

    # Also create a function that samples a certain number of random images from the dataset and then encodes / decodes
    # them
    def reconstruct_random_sample():

        # Grab num_random_samples random samples from the dataset
        random_ground_truths = dataset.data[torch.randint(low=0, high=len(dataset.data), size=(num_random_samples,))]

        # Put the ground truths on the right device
        random_ground_truths = random_ground_truths.to(device)

        # Encode and decode them
        reconstructions = decoder(encoder(random_ground_truths))

        # Return the reconstructions and the ground truths
        return reconstructions, random_ground_truths

    # Define the list / range of epochs at which to save the model
    checkpoints = list(
        range(
            snapshot_frequency,
            num_epochs + 1,
            snapshot_frequency,
        )
    )

    # Create a specs file where we remember / save all the options that we used
    specs = {
        "experiment_directory": experiment_directory,
        "prev_model_dir": prev_model_dir,
        "latent_dim": latent_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "number_of_classes": number_of_classes,
        "beta": beta,
        "alpha": alpha,
        "log_frequency": log_frequency,
        "snapshot_frequency": snapshot_frequency,
        "num_random_samples": num_random_samples,
        "batch_size_update_freq": batch_size_update_freq,
        "max_batch_size": max_batch_size
    }

    # Save the specs file in the experiment directory
    with open(os.path.join(experiment_directory, "specs.json"), "w") as f:
        json.dump(specs, f)

    # Create a weights and biases (WB) session where we monitor the losses and also save the specifications
    experiment_name = os.path.basename(experiment_directory)
    wandb.init(project="DL-Circ-Tumour-Cells", dir=os.path.join(experiment_directory),
               name=os.path.dirname(experiment_name), reinit=True, config=specs, resume=True,
               notes="The experiment directory is: {}".format(experiment_name))#, mode="disabled")

    # Create a counter that keeps track how many batches we have dealt with
    batch_counter = 0

    # Start training
    for epoch in range(num_epochs):

        print("Current epoch: {}/{}".format(epoch, num_epochs))

        # After every ... epochs, overwrite the dataloader with a dataloader that has an increases batch_size
        if epoch % batch_size_update_freq == 0:
            batch_size = min(2 * batch_size, max_batch_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Grab a batch from the dataloader
        for images, labels in dataloader:

            # Increase the batch counter
            batch_counter += 1

            # Put the data on the correct device
            images, labels = images.to(device), labels.to(device)

            # Reset the optimizer
            optimizer.zero_grad()

            # Put this batch in the encoder
            latent_codes = encoder(images)

            # Get the reconstructions
            reconstructions = decoder(latent_codes)

            # Get the classification probabilities
            probs = classifier(latent_codes)

            # Flatten the reconstructions and the ground truth images such that we can use the cross entropy
            # reconstruction loss
            reconstructions_flattened = reconstructions.flatten(start_dim=1, end_dim=-1)
            images_flattened = images.flatten(start_dim=1, end_dim=-1)

            # Calculate the reconstruction loss
            #recon_loss = torch.mean(torch.sum((reconstructions_flattened - images_flattened) ** 2, dim=-1))
            recon_loss = binary_cross_entropy_loss(reconstructions_flattened, images_flattened) * 80 * 80 * 3

            # Calculate the classification loss
            classification_loss = negative_log_likelihood(torch.log(probs), labels)

            # Calculate the regularization on the latent codes
            lat_code_reg = torch.mean(torch.sum(latent_codes ** 2, dim=-1))
            #lat_code_reg = torch.sum(latent_codes ** 2) # This is the one used in the original code I think, but I do not think it is a good choice!

            # Calculate the full loss
            loss = gamma * recon_loss + beta * classification_loss + alpha * lat_code_reg

            # Backpropagate the gradients
            loss.backward()

            # Perform an optimization step using the optimizer
            optimizer.step()

            # Save the losses
            wandb.log({'recon_loss': recon_loss.item(), 'classification_loss': classification_loss.item(),
                       'latent_code_loss': lat_code_reg.item(), 'full_loss': loss.item(), 'beta': beta},
                      step=batch_counter)

        # Saving the models on the indicated checkpoints. Also save some images of ground truth images and the
        # corresponding reconstructions side-by-side.
        if epoch in checkpoints:

            # Save the model
            save_checkpoints(epoch)

            # Get some random ground truths and corresponding reconstructions and make sure the resulting tensors are
            # of shape 'batch_size x width x height x 3'. This is needed for the function creating and saving some
            # figures.
            reconstructions, ground_truths = reconstruct_random_sample()
            reconstructions = reconstructions.permute((0, 2, 3, 1)).detach().cpu().numpy()
            ground_truths = ground_truths.permute((0, 2, 3, 1)).detach().cpu().numpy()

            # Create and save figures that put the reconstruction and the ground truth image side-by-side.
            save_reconstruction_and_gt_images(os.path.join(experiment_directory, "figures", str(epoch)),
                                              reconstructions, ground_truths)

        # Save the model after every specific amount of epochs. Also save some images of ground truth images and the
        # corresponding reconstructions side-by-side.
        if epoch % log_frequency == 0 or epoch == (num_epochs-1):

            # Save the model
            save_latest(epoch)

            # Get some random ground truths and corresponding reconstructions and make sure the resulting tensors are
            # of shape 'batch_size x width x height x 3'. This is needed for the function creating and saving some
            # figures.
            reconstructions, ground_truths = reconstruct_random_sample()
            reconstructions = reconstructions.permute((0, 2, 3, 1)).detach().cpu().numpy()
            ground_truths = ground_truths.permute((0, 2, 3, 1)).detach().cpu().numpy()

            # Create and save figures that put the reconstruction and the ground truth image side-by-side.
            save_reconstruction_and_gt_images(os.path.join(experiment_directory, "figures", 'latest'),
                                              reconstructions, ground_truths)


def multiple_train_loops(beta_list, experiment_directory, latent_dim=50, num_epochs=75, batch_size=500,
                         number_of_classes=2, alpha=0.001, gamma=1.0, log_frequency=100, snapshot_frequency=100,
                         num_random_samples=3, batch_size_update_freq=125, max_batch_size=256):

    # Initialize prev_model_dir. This variable keeps track of where the last trained model is stored.
    prev_model_dir = None

    # For every beta parameter in the list of beta parameters, do ...:
    for beta in beta_list:

        # Define the right subdirectory in the main directory
        sub_experiment_directory = os.path.join(experiment_directory, "model_beta_{}".format(beta))

        # Create it if it does not exist
        if not os.path.isdir(sub_experiment_directory):
            os.makedirs(sub_experiment_directory)

        # Train the model with the specific beta value
        train(sub_experiment_directory, prev_model_dir, latent_dim=latent_dim, num_epochs=num_epochs,
              batch_size=batch_size, number_of_classes=number_of_classes, beta=beta, alpha=alpha, gamma=gamma,
              log_frequency=log_frequency, snapshot_frequency=snapshot_frequency, num_random_samples=num_random_samples,
              batch_size_update_freq=batch_size_update_freq, max_batch_size=max_batch_size)

        # Update the prev_model_dir
        prev_model_dir = sub_experiment_directory


if __name__ == "__main__":
    multiple_train_loops(beta_list=(0, 10, 100, 1000), experiment_directory=os.path.join("..", "results", "test"),
                         latent_dim=50, num_epochs=75, batch_size=16, number_of_classes=6, alpha=0.01, gamma=1.0,
                         log_frequency=25, snapshot_frequency=25, num_random_samples=3, batch_size_update_freq=15,
                         max_batch_size=256)
