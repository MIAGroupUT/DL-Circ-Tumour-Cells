# Deep learning of circulating tumour cells

This repository tries to reproduce the results in the paper <ins> [Deep learning of circulating tumour cells](https://www.nature.com/articles/s42256-020-0153-x) </ins>
by Zeune et al. 

NOTE: this repository is only tested on Linux.

## Training the model(s)
The general training paradigm trains a model by training several models subsequently. The previously trained model is used as initialization of the 'new' model. Moreover, for every 'new' model, we only change the 'beta' constant of the classification loss. The other parameters are reset to their initial value. These initial values are saved in a *specs_base.json* file. For every experiment you do, you should create a specific directory where you store this the *specs_base.json* file. This file can, for instance, be:
```
{
    "beta_list": [0.0, 10.0, 100.0, 1000.0],
    "latent_dim": 50,
    "num_epochs": 75,
    "batch_size": 16,
    "number_of_classes": 6,
    "alpha": 0.01,
    "gamma": 1.0,
    "log_frequency": 25,
    "snapshot_frequency": 25,
    "num_random_samples": 3,
    "batch_size_update_freq": 15,
    "max_batch_size": 256
}
```
Here:
- **beta_list**: we train the model ``len(beta_list)`` times where the i-th run uses parameter `beta_list[i-1]` as **beta** and uses the (i-1)st trained model as initialization. 
- **latent_dim**: the latent dimension used in the model
- **batch_size**: the initial batch size used during training
- **number_of_classes**: there is a dataset that has the images divided into two classes (/labels), one that divides them into 5 classes, and one that divides them into 6 classes. The **number_of_classes** variable determines which dataset we use.
- **alpha**: the latent code regularization loss coefficient
- **gamma**: the constant in front of the reconstruction loss
- **log_frequency**: the number of epochs after which to save the LATEST version of the model. NOTE: this overwrites the previous LATEST version.
- **snapshot_frequency**: the number of epochs after which to save the currently trained model. NOTE: this does not overwrite the previous LATEST version of the model but creates a different file for the model at that specific epoch.
- **num_random_samples**: the number of samples used to make some figures of the reconstructions.
- **batch_size_update_freq**: the number of epochs after which we multiply the batch size by 2
- **max_batch_size**: the maximum batch size possible

NOTE: for training the i-th model, all the above parameters are reinitialized. So, e.g. we start with the initial batch size again and do not continue with the batch size that the previous training ended with.

NOTE: each model trained with a specific **beta** parameter will get its own directory where e.g. the snapshots and reconstruction figures are saved.

To train the model, you can execute the following command in the command line

```
python train.py -e experiment_directory
```
Here *experiment_directory* is the earlier mentioned directory that contains the *specs_base.json* file.

After training, the folder will look like:

```
experiment_directory
│
└─── model_beta_{beta_list[0]}
│   │
│   └─── figures
│   │   │
│   │   └─── n_0
│   │   │   │
│   │   │   └─── Reconstruction_vs_GT_pair_*.png (the * indicates that there might be multiple of such figures)
│   │   │
│   │   └─── n_1
│   │   │   │
│   │   │   └─── Reconstruction_vs_GT_pair_*.png
│   │  ...
│   │   │
│   │   └─── latest     
│   │       │
│   │       └─── Reconstruction_vs_GT_pair_*.png      
│   │
│   └─── ModelParameters
│   │   │   
│   │   └─── classifier
│   │   │   │
│   │   │   └─── n_0.pth, n_1.pth, ..., latest.pth
│   │   │
│   │   └─── decoder
│   │   │   │
│   │   │   └─── n_0.pth, n_1.pth, ..., latest.pth
│   │   │
│   │   └─── encoder     
│   │       │
│   │       └─── n_0.pth, n_1.pth, ..., latest.pth   
│   │
│   └─── OptimizerParameters
│   │   │   
│   │   └─── n_0.pth, n_1.pth, ..., latest.pth
│   │
│   └─── wandb
│   │
│   └─── specs.json
│
└─── model_beta_{beta_list[1]}
│   │
│  ...
│
...
│
│
└─── specs_base.json
```
where:
- **model_beta_{beta_list[i-1]}**: the subfolder containing all the relevant information regarding the i-th trained model.
- **n_0, n_1, ...**: the specific epochs at which we saved the model.
- **latest**: points to reconstruction figures, model parameters, and optimizer parameters of the last saved model.
- **figures**: contains figures showing the reconstruction vs. the ground truth image.
- **ModelParameters**: each subfolder contains parameters of the specific model that the subfolder corresponds to.
- **OptimizerParameters**: saves the state of the optimizer at the moment of saving the model.
- **wandb**: a folder corresponding to the used 'weights and biases' session.
- **specs.json**: a specs.json file containing the specific parameters used for training a specific model. This will be used in the evaluation code.

## Evaluating the trained models

To evaluate a trained model, you can execute the following command in the command line

```
python evaluation.py -m model_path -b_x_l -3 -b_x_r 7 -b_y_b 2 -b_y_t 13
```
To see what *evaluation.py* does, see the documentation of the *evaluate_model_on_data* function in *evaluation.py*.

In the above bash script:
- **model_path**: points to the directory of the model we want to evaluate. Could e.g. be 'experiment_directory/model_beta_{beta_list[-1]}'.
- **-b_x_***: denotes the x-value of the left-side, the x-value of the right-side, the y-value of the bottom-side, and the y-value of the top side of a box in R^2, respectively. This box is used to plot reconstructions of points within that box in a tSNE-plot.
