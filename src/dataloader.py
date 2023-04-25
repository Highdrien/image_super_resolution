import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

np.random.seed(0)


class DataGenerator(Dataset):
    def __init__(self, config, mode):
        self.mode = mode
        print('mode:', mode)
        if mode == 'train':
            self.path = config.train.path
        elif mode == 'val':
            self.path = config.train.path
        elif mode == 'test':
            self.path = config.train.path
        else:
            print('mode is not valid')
            exit()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.labels_path)

    def __getitem__(self, index):
        """
        get the hr (high resolution), x2, x3, x4 from a image number (index)
        """
        image_name = str(index).zfill(4)    # to convert xxx to '0xxx' or xx to '00xx'
        hr = read_image(os.path.join(self.path.HR, image_name + '.png'))
        x2 = read_image(os.path.join(self.path.X2, image_name + 'x2.png'))
        x3 = read_image(os.path.join(self.path.X3, image_name + 'x3.png'))
        x4 = read_image(os.path.join(self.path.X4, image_name + 'x4.png'))
        return hr, x2, x3, x4


def plot_getitems(images):
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        image = images[i]
        image = np.moveaxis(image.numpy(), 0, -1)
        axes[i // 2, i % 2].imshow(image)

    plt.show()


def create_generator(config, mode):
    """Returns generator from a config and mode ('train','val','test')"""
    test = DataGenerator(config, mode)
    images = test.__getitem__(235)
    plot_getitems(images)

    # DataLoader(DataGenerator(train_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
