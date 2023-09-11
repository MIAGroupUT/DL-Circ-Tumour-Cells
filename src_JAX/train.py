from src.models import Encoder, Decoder, Classifier
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomAffine, InterpolationMode



def train(latent_dim=50, num_epochs=75):

    # First, create the different models
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    classifier = Classifier(latent_dim)

    # Load the dataset
    dataset = ...

    # Define the transformations used for data augmentation
    transforms = Compose([
        RandomHorizontalFlip(0.5),
        # Zoomout still needed and the scaling between 0 and 1 is still needed. The customized background substraction
        # as well.
        RandomAffine(40, translate=(0.15, 0.15), shear=0.05, interpolation=InterpolationMode.NEAREST)
    ])

    # Get the dataloader
    dataloader = ...

    # Define the optimizers
    optimizer = ...

    # Start training
    for epoch in num_epochs:

        # Grab a batch from the dataloader
        for images, labels in dataloader:

            # Reset the optimizer
            optimizer.zero_grad()

            # Put this batch in the encoder
            latent_codes = encoder(images)

            # Get the reconstructions
            reconstructions = decoder(latent_codes)

            # Get the classification probabilities
            probs = classifier(latent_codes)

            # Calculate the loss function
            loss = ...

            # Backpropagate the gradients
            loss.backward()

            # Perform an optimization step using the optimizer
            optimizer.step()
