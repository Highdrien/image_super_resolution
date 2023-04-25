import os
import numpy as np

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
            self.path = config.train.labels_path
            self.labels_name = os.listdir(config.train.labels_path)
        elif mode == 'val':
            self.path = config.val.labels_path
            self.labels_name = os.listdir(config.val.labels_path)
        elif mode == 'test':
            self.path = config.test.labels_path
            self.labels_name = os.listdir(config.test.labels_path)
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
        Generate one batch of data
        """
        image_path = os.path.join(self.path, self.labels_name[index])
        y = read_image(image_path)
        print(y.shape)
        return y



def create_generator(config, mode):
    """Returns generator from a config and mode ('train','val','test')"""
    test = DataGenerator(config, mode)
    test.__getitem__(0)

    # DataLoader(DataGenerator(train_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
