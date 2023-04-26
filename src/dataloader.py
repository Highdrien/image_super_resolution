import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

np.random.seed(0)


class DataGenerator(Dataset):
    def __init__(self, config, mode, upscale_factor):
        self.mode = mode
        self.upscale_factor = upscale_factor

        # path to datas
        self.HR_path = config[mode].path.HR
        self.LR_path = config[mode].path['X' + str(upscale_factor)]

        # end of images name
        self.HR_end = '.png'
        self.LR_end = 'x' + str(upscale_factor) + '.png'

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.labels_path)

    def __getitem__(self, index):
        """
        get the hr (high resolution), x2, x3, x4 from a image number (index)
        """
        image_name = str(index + 1).zfill(4)    # to convert xxx to '0xxx', xx to '00xx', x to '000x'

        hr_image = read_image(os.path.join(self.HR_path, image_name + self.HR_end))
        lr_image = read_image(os.path.join(self.LR_path, image_name + self.LR_end))

        return hr_image, lr_image


def plot_getitem(images):
    fig, axes = plt.subplots(1, 2)
    for i in range(2):
        image = images[i]
        image = np.moveaxis(image.numpy(), 0, -1)
        axes[i].imshow(image)

    plt.show()


def plot_getitem_with_zoom(images, zoom, upscale_factor):
    _, axes = plt.subplots(2, 2)
    titles = ["HR image", "upscale: "+str(upscale_factor), "zoom x"+str(zoom), "zoom x"+str(zoom)]
    for i in range(2):
        image = images[i]
        image = np.moveaxis(image.numpy(), 0, -1)
        axes[0, i].imshow(image)
        axes[0, i].set_title(titles[i], y=-0.3)
        w, h = image.shape[:-1]
        zoom_image = image[w//2:w//2+w//zoom, h//2:h//2+h//zoom, :]
        axes[1, i].imshow(zoom_image)
        axes[1, i].set_title(titles[2 + i], y=-0.3)

    plt.show()


def create_generator(config, mode):
    """Returns generator from a config and mode ('train','val','test')"""
    upscale_factor = 4
    train_generator = DataGenerator(config, mode, upscale_factor)
    images = train_generator.__getitem__(2)
    # plot_getitem(images)
    plot_getitem_with_zoom(images, 15, upscale_factor)

    # DataLoader(DataGenerator(train_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
