# Image Super Resolution

# Description
Super resolution from a single low-resolution image has always been an ill posed problem. The current state of the art consists in training a convolutional neural network with a large dataset of natural looking images. Corresponding low-resolution images are computed and fed as an input to the neural network. The learned feature maps have shown to produce less artifacts than classical methods.

# Table of Contents
- [Image Super Resolution](#image-super-resolution)
- [Description](#description)
- [Table of Contents](#table-of-contents)
- [Objectives](#objectives)
- [Data](#data)
  - [the dataset](#the-dataset)
  - [The data is then arranged as follows:](#the-data-is-then-arranged-as-follows)
- [Requirements](#requirements)
- [Run the code](#run-the-code)
  - [Mode: train](#mode-train)
  - [Mode: resume train](#mode-resume-train)
  - [Mode: tranfer learning](#mode-tranfer-learning)
  - [Mode: test](#mode-test)
  - [Mode: predict](#mode-predict)
  - [Mode: bicubic](#mode-bicubic)
  - [Mode: predict bicubic](#mode-predict-bicubic)
- [Experiment](#experiment)
  - [experiment 1: upscale factor = 3](#experiment-1-upscale-factor--3)
  - [experiment 2: upscale factor = 3](#experiment-2-upscale-factor--3)
  - [experiment 3: upscale factor = 2](#experiment-3-upscale-factor--2)
  - [experiment 4: upscale factor = 4](#experiment-4-upscale-factor--4)
- [Comparaison with bicubic](#comparaison-with-bicubic)


# Objectives
The aim of this project is to build a neural network that increases the number of pixels in
images. We would like to have several models that increase the length and width of images
by a factor of 2, 3 and 4. In the following we will call this factor the upscale factor. Our
aim is of course to get the best performance, but more concretely, we want our network to
perform better than the bicubic method. The bicubic method consists of increasing the
number of pixels simply by adding the sames pixels. For example, if we want to increase
an image by an upscale factor of 2, each pixel of the image will be transformed into a
square of 2 ∗ 2 pixels of the same value. This increases the number of pixels in the image.
However, this method is not the most effective because when you zoom in on an image,
these squares appear. That’s why we want to propose a more effective method than the
bicubic.

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

To run the code you need python (We use python 3.9.13) and packages in the following versions :

- torch==2.0.0
- numpy==1.23.0
- torchvision==0.15.1
- easydict==1.10
- PyYAML==6.0
- matplotlib==3.6.2
- tqdm==4.64.1
- torchmetrics==0.11.4
- opencv-python==4.7.0.72

You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Run the code

To run the program, simply execute the `main.py` file. However, there are several modes.

## Mode: train

To do this, you need to choose a `.yaml` configuration file to set all the training parameters. By default, the code will use the `config/configs.yaml` file. The code will create a folder: 'experiment' in logs to store all the training information, such as a copy of the configuration used, the loss and metrics values at each epoch, the learning curves and the model weights.
To run a training session, enter the following command:
```bash
python main.py --mode train --config_path <path to your configuration system> 
```

## Mode: resume train

The resumetrain mode is used to continue a training session. All you need to do is select the path of the experiment you want to continue, and execute the command:
```bash
python main.py --mode resumetrain --path <path to the experiment which you want to resume> 
```

## Mode: tranfer learning

The tranfer leanrning mode allows you to create a new experiment from a completed experiment and to perform transfer learning to train a model on another upscale factor. You will need to execute the following command:
```bash
python main.py --mode tf --path <path to the completed experiment> --new_upscale_factor <you upscale factor> 
```

## Mode: test

Test mode is used to evaluate an experiment on the test database. It will give you the value of the Loss and the metrics in a file test_log.txt which will be located in the experiment folder.
```bash
python main.py --mode test --path <path to the experiment which you want to evaluate> 
```

## Mode: predict

The predict mode is used to predict all the images in the congif.predict.src_path folder. These images will be split into patches, then predicted, then reconstituted and finally saved in the folder indicated in config.predict.dst_path.

```bash
python main.py --mode predict --path <path to the experiment which you want to use for the prediction> 
```

## Mode: bicubic

The bicubic mode is used to calculate the MSE and metrics for upscale factor 2, 3 and 4. The values are stored in the logs/bicubic/test_logs.txt file. You can preset a configuration to indicate which metrics to use.
```bash
python main.py --mode bicubic  --config_path <path to your configuration system> 
```

## Mode: predict bicubic

The predict bicubic mode is used to predict all the images in the congif.predict.src_path folder with the bicubic method. These images will be saved in the folder indicated in config.predict.dst_path.

```bash
python main.py --mode predict_bicubic
```

# Experiment

we ran several experiments that you can find in the  `logs` file:

## experiment 1: upscale factor = 3
`experiment_1`: training on all the patches with 15 epochs (with hidden_channels_1 = 64)

<p align="center"><img src=logs/experiment_1/MSE.png><p>

After a test, we find:

MSE = 0.0022520951427679485\
PSNR = 26.536483545971524\
MSSSIM = 0.9671353705369743

## experiment 2: upscale factor = 3
`experiment_2`: training on all the patches with 30 epochs (with hidden_channels_1 = 128)

<p align="center"><img src=logs/experiment_2/MSE.png><p>

After a test, we find:

MSE = 0.0019440386436173729\
PSNR = 27.189252033355128\
MSSSIM = 0.9738962206111592

## experiment 3: upscale factor = 2
`experiment_3`: tranfer learning from the experiment 2 to change the upscale factor. There are only 2 epoches because it's was enough. After the test, we find: 

MSE = 0.0012848700122732764\
PSNR = 28.97259296125667\
MSSSIM = 0.9879952942489818

## experiment 4: upscale factor = 4
`experiment_4`: tranfer learning from the experiment 2 to change the upscale factor. There are only 2 epoches because it's was enough. After the test, we find: 

MSE: 0.003155684844919949\
PSNR: 25.071385717695687\
MSSSIM: 0.9489965659038276


# Comparaison with bicubic

| upscale factor =2 | MSE | PSNR | MSSSIM |
| :---:   |:-------: | :---:   | :---:   |
| bicubic | 0.00189 | 28.39 | 0.991 |
| experiment 3| 0.00128 | 28.97 | 0.987 |

| upscale factor = 3 | MSE | PSNR | MSSSIM |
| :---:   |:-------: | :---:   | :---:   |
| bicubic | 0.00322 | 25.97 | 0.958 |
|experiment 1 | 0.00225 | 26.53 | 0.9671|
| experiment 2| 0.00194 | 27.18 | 0.973 |

| upscale factor = 4 | MSE | PSNR | MSSSIM |
| :---:   |:-------: | :---:   | :---:   |
| bicubic | 0.00422 | 24.70 | 0.936 |
| experiment 4| 0.00315 | 25.07 | 0.948 |
