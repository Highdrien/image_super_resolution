import os

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from src.utils import plot_getitem

torch.manual_seed(0)


class DataGenerator(Dataset):
    """
    generator for train, validation and test
    """
    def __init__(self, config, mode):
        if mode not in ['train','val', 'test']:
            print("invalid mode: choose between train, val and test")
            exit()

        self.mode = mode
        self.upscale_factor = config.upscale_factor
        self.normalisation = config.model.data_normalisation

        # path to datas
        self.HR_path = os.path.join(config.data.path, config[mode].path.HR)
        self.LR_path = os.path.join(config.data.path, config[mode].path['X' + str(self.upscale_factor)])

        # list of image name
        self.HR_data = os.listdir(self.HR_path)
        self.LR_data = os.listdir(self.LR_path)

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.LR_data)

    def __getitem__(self, index):
        """
        get the hr (high resolution) and the lr (low resolution) from a image number (index)
        """
        hr_image = read_image(os.path.join(self.HR_path, self.HR_data[index]))
        lr_image = read_image(os.path.join(self.LR_path, self.LR_data[index]))
        
        if self.normalisation:
            lr_image = lr_image / 255
            hr_image = hr_image / 255

        return lr_image, hr_image


def create_generator(config, mode):
    """Returns generator from a config and a mode ('train','val','test')"""
    generator = DataGenerator(config, mode)
    return DataLoader(generator, 
                      batch_size=config[mode].batch_size, 
                      shuffle=config[mode].shuffle, 
                      drop_last=config[mode].drop_last)


def getbatch(config, mode):
    dataloader = create_generator(config, mode)
    X, Y = next(iter(dataloader))
    plot_getitem(X, Y, config.upscale_factor, index=2)



class PredictGenerator(Dataset):
    """
    generator for the prediction
    """
    def __init__(self, config):

        self.upscale_factor = config.upscale_factor
        self.normalisation = config.model.data_normalisation

        # path to datas
        self.src_path = config.predict.src_path

        # list of image name
        self.images_name = os.listdir(self.src_path)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        """
        get the lr (low resolution) from a image number (index)
        """
        lr_image = read_image(os.path.join(self.LR_path, self.LR_data[index]))
        
        if self.normalisation:
            lr_image = lr_image / 255

        return lr_image


def create_predict_generator(config):
    """Return the prediction's generator from a config """
    generator = PredictGenerator(config)
    return DataLoader(generator, 
                      batch_size=config.predict.batch_size, 
                      shuffle=False, 
                      drop_last=False)