from src.models import Encoder, Decoder, Classifier
from src.data import Dataset

from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomAffine, InterpolationMode
from torch.utils.data.dataloader import DataLoader

import torch

cross_entropy_loss = torch.nn.CrossEntropyLoss()

binary_cross_entropy_loss = torch.nn.BCELoss(reduction='mean')


def train(latent_dim=50, num_epochs=75, batch_size=500, train_or_val='train', number_of_classes=2, beta=1.0):

    # Use cuda if cuda is available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # First, create the different models
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    classifier = Classifier(latent_dim, number_of_classes).to(device)

    # Define the transformations used for data augmentation
    transforms = Compose([
        RandomHorizontalFlip(0.5),
        RandomAffine(40, translate=(0.15, 0.15), shear=0.05, scale=(1-0.2, 1+0.2), interpolation=InterpolationMode.NEAREST)
    ])

    transforms = None

    # TODO: make sure the transforms above correspond to the ones used in the paper.

    # Load the dataset
    dataset = Dataset(train_or_val, number_of_classes, transforms)

    # Get the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define the optimizers

    # optimizer = torch.optim.Adadelta(list(encoder.parameters()) + list(decoder.parameters()) +
    #                                  list(classifier.parameters()), lr=0.001)

    optimizer = torch.optim.Adadelta(
        [
            {
                "params": encoder.parameters(),
                "lr": 0.001,
            },
            {
                "params": decoder.parameters(),
                "lr": 0.001,
            },
            {
                "params": classifier.parameters(),
                "lr": 0.001,
            },
        ]
    )

    # TODO: check if the adadelta parameters are as in the paper

    # Start training
    for epoch in range(num_epochs):

        print("Current epoch: {}/{}".format(epoch, num_epochs))

        # After every ... epochs, overwrite the dataloader with a dataloader that has an increases batch_size
        if epoch % 125 == 0:
            batch_size = 2 * batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Grab a batch from the dataloader
        for images, labels in dataloader:

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

            # TODO: check if the loss function below is correctly implemented

            # Calculate the loss function
            # loss = torch.mean(torch.sum((reconstructions_flattened - images_flattened) ** 2, dim=-1))

            loss = binary_cross_entropy_loss(reconstructions_flattened, images_flattened) * 80 * 80 * 3

            # loss = binary_cross_entropy_loss(reconstructions_flattened, images_flattened) + \
            #        beta * cross_entropy_loss(probs, labels)

            #loss = binary_cross_entropy_loss(reconstructions.flatten(), images.flatten()) * 80 * 80 * 3 + \
            #       beta * cross_entropy_loss(probs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Perform an optimization step using the optimizer
            optimizer.step()

        if epoch % 100 == 0:
            import matplotlib.pyplot as plt

            for i in range(min(batch_size, 3)):

                fig, (ax1, ax2) = plt.subplots(1, 2)

                t1 = reconstructions[i].permute((1, 2, 0)).detach().cpu().numpy()
                t2 = images[i].permute((1, 2, 0)).detach().cpu().numpy()

                ax1.imshow(t1)
                ax2.imshow(t2)
                plt.show()
                # plt.imshow(t1)
                # plt.show()
                #
                # plt.imshow(t2)
                # plt.show()

        print(loss.item())

if __name__ == "__main__":
    train(num_epochs=500, batch_size=16, beta=0.0)
