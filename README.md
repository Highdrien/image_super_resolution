# Image Super Resolution

# Description
Super resolution from a single low-resolution image has always been an ill posed problem. The current state of the art consists in training a convolutional neural network with a large dataset of natural looking images. Corresponding low-resolution images are computed and fed as an input to the neural network. The learned feature maps have shown to produce less artifacts than classical methods.

# Objectives
The first step in this project is to retrieve the dataset. We suggest using the [DIVerse 2K resolution high quality images](https://data.vision.ee.ethz.ch/cvl/DIV2K/) which provides both high- and low-resolution images. The most interesting aspect of this project is the method to upscale the number of pixels in a given layer of your network. A very powerful approach is to interleave pixels of separate feature maps in a so-called sub-pixel convolution. Other works work on a bicubic interpolation of the low-resolution image to avoid this, or use sparse-coding to map low-resolution patches to high-resolution ones. After training, compute the PSNR of the reconstructed image on a validation dataset. The deliverables of this project are: 
- a brief description of the literature ; 
- a working super resolution CNN with a few different architecture parameters; 
- an experimental study of the reconstructed image PSNR against plain bicubic interpolation.

# Data

## the dataset

For this project we used images from the DIV2K dataset. if you want them, go to their website: https://data.vision.ee.ethz.ch/cvl/DIV2K/ and download the following folders:
(NTIRE 2017) Low Res Images:
- Train Data Track 1 bicubic downscaling x2 (LR images)
- Validation Data Track 1 bicubic downscaling x2 (LR images)
- Train Data Track 1 bicubic downscaling x3 (LR images)
- Validation Data Track 1 bicubic downscaling x3 (LR images)
- Train Data Track 1 bicubic downscaling x4 (LR images)
- Validation Data Track 1 bicubic downscaling x4 (LR images)
High Resolution Images:
- Train Data (HR images)
- Validation Data (HR images)


## The data is then arranged as follows:

In the `data` folder, you will find all the images used in the training. It contains:
- `DIV2K`: a folder with the initial images. It contains the following folders:
  - `DIV2K_train_HR`: folder containing all the training images in high definition (2K)
  - `DIV2K_train_LR_bicubic`: corresponding low resolution images obtained using Matlab imresize function with default settings (bicubic interpolation), which containts:
    - `X2`: LR images, downscale factor 2
    - `X3`: LR images, downscale factor 3
    - `X4`: LR images, downscale factor 4\
  and the same for valid (validation data)

- `patches`: which are image patches all of the same size that come from the DIV2K images. To get them, you have to run the code `create_patches.py`.
- `create_patches.py`: To run it, go to the data folder, and run the code



# Requirements

To run the code you need python (I use python 3.9.13) and packages in the following versions :

- torch==2.0.0
- numpy==1.23.0
- torchvision==0.15.1
- easydict==1.10
- PyYAML==6.0
- matplotlib==3.6.2
- tqdm==4.64.1

You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Run the code

You can run the script with the following command:
```bash
python main.py --mode <mode> --config_path <path to your config>
```

- `mode`: with the mode, you will choose between to run the code to do a train, a test or a predict.
-  `config_path`: if you want to train the model, you must specify the parameters used for the training. To do this you need to specify the path to a yaml file which containe all the parameters. If you don't specify it, it will take the file `config.yaml` which is in `config` folder

# Results

we ran several experiments that you can find in the  `logs` file:

- `experiment_0`: training on few patches (one patche with size 120*120 per image) to see the result

<p align="center"><img src=logs/experiment_0/MSE.png><p>