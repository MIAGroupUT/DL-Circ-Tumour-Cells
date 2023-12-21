# Deep learning of circulating tumour cells

This repository reproduces the results in the paper <ins> [Deep learning of circulating tumour cells](https://www.nature.com/articles/s42256-020-0153-x) </ins>
by Zeune et al.

## Installation
The packages used and their specific versions are:
```
python=3.11.5
torch=2.1.2
torchvision=0.16.2
torchaudio=2.1.2
pytorch-cuda=12.1
matplotlib=3.8.0
scikit-learn=1.2.2
pandas=2.1.4
seaborn=0.13.0
wandb=0.16.1
```
One can also use the `environment.yml` file to create a conda environment that contains the required packages.

## Training the model(s) 
The general training paradigm trains a model by training several models subsequently. The previously trained model is used as initialization of the 'new' model. Moreover, for every 'new' model, we only change the 'beta' constant of the classification loss. The other parameters are reset to their initial value. These initial values are saved in a `specs_base.json` file. For every experiment you do, you should create a specific directory where you store this `specs_base.json` file. 

An example `specs_base.json` file is given in the `results/cell_data` directory. To run the code using this `specs_base.json` file, execute the following command in the command line (after activating the correct Python/Conda environment):
```
python train.py -e results/cell_data
```
Here we assume we run the code from the main directory of the repository. 

## The required options in `specs_base.json`
The `specs_base.json` file has the following options:
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

## Evaluating the trained models

To evaluate a trained model, you can execute the following command in the command line

```
python evaluation.py -m model_path -b_x_l -3 -b_x_r 7 -b_y_b 2 -b_y_t 13
```
To see what `evaluation.py` does, see the documentation of the `evaluate_model_on_data` function in `evaluation.py`.

In the above bash script:
- **model_path**: points to the directory of the model we want to evaluate. Could e.g. be `results/cell_data/model_beta_{beta_list[i]}`. Here `model_beta_{beta_list[i]}` is the folder containing, among others, the weights of the (i+1)-st trained model. For more info on these folders, see the next section.  
- **-b_\*_***: denotes the x-value of the left-side, the x-value of the right-side, the y-value of the bottom-side, and the y-value of the top side of a box in R^2, respectively. For points in a tSNE-plot that are inside this box, we plot their reconstructed images.

## Folder structure after training
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
- **latest**: points to reconstruction figures, model weights, and optimizer parameters of the last saved model.
- **figures**: contains figures showing the reconstruction vs. the ground truth image.
- **ModelParameters**: each subfolder contains the weights of the specific neural network model that the subfolder corresponds to.
- **OptimizerParameters**: saves the state of the optimizer at the moment of saving the model.
- **wandb**: a folder corresponding to the used 'weights and biases' session.
- **specs.json**: a `specs.json` file containing the specific parameters used for training a specific model. This will be used in the evaluation code.

## Some final remarks
Here we provide some comments regarding the code:
- The code has only been tested on Linux. If you encounter problems on Windows, we suggest using Windows Subsystem for Linux (WSL2). 
- For training the i-th model, all specified parameters in `specs_base.json` are reinitialized. For example, we start with the initial batch size again and do not continue with the batch size that the previous training loop ended with.

## Citation
When using this code, please reference the following paper:
```
@article{zeune2020deep,
  title={Deep learning of circulating tumour cells},
  author={Zeune, Leonie L and Boink, Yoeri E and van Dalum, Guus and Nanou, Afroditi and de Wit, Sanne and Andree, Kiki C and Swennenhuis, Joost F and van Gils, Stephan A and Terstappen, Leon WMM and Brune, Christoph},
  journal={Nature Machine Intelligence},
  volume={2},
  number={2},
  pages={124--133},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
```

## Contact
This repository is developed by Sven Dummer and is a part of his PhD. When you have any questions regarding the code, do not hesitate to:
- Open an issue,
- Or contact Sven by sending an email to [s.c.dummer@utwente.nl](mailto:s.c.dummer@utwente.nl)

