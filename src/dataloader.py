import os
import math
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

        self.size_patches = config.data.image_size
        self.size_overlay = config.predict.size_overlay

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        """
        get the lr (low resolution) from a image number (index)
        """
        lr_image = read_image(os.path.join(self.src_path, self.images_name[index]))
        shape = lr_image.shape
        size_overlay = self.size_overlay
        size_patches = self.size_patches
        if self.normalisation:
            lr_image = lr_image / 255

        canal, x, y = lr_image.shape

        # Calculate the number of patches in each dimension
        nb_patches_x = math.ceil((x-size_patches)/(size_patches-size_overlay))+1
        nb_patches_y = math.ceil((y-size_patches)/(size_patches-size_overlay))+1
        nb_patches = nb_patches_x * nb_patches_y

        # Calculate the modified image dimensions
        # x_modifie = x + (self.size_patches - x % self.size_patches) if x % self.size_patches != 0 else x
        # y_modifie = y + (self.size_patches - y % self.size_patches) if y % self.size_patches != 0 else y
        x_modifie = size_patches+(size_patches-size_overlay)*(nb_patches_x-1)
        y_modifie = size_patches+(size_patches-size_overlay)*(nb_patches_y-1)
        
        # Create the tensor for the modified image
        image_modifiee = torch.zeros((canal, x_modifie, y_modifie))

        # Copy the original image in the modified image
        image_modifiee[:, :x, :y] = lr_image

        c, w, h = image_modifiee.shape

        # nb_patches_x = w // self.size_patches
        # nb_patches_y = h // self.size_patches



        # Create torsor to save patches
        patches = torch.zeros((nb_patches, c, self.size_patches, self.size_patches))

        # Cutting the image into patches
        # idx = 0
        for i in range(nb_patches_x):
            for j in range(nb_patches_y):                
                # begin_x = i * self.size_patches
                # begin_y = j * self.size_patches
                # fin_x = begin_x + self.size_patches #begin in english and fin in french XD
                # fin_y = begin_y + self.size_patches
                # sous_image = image_modifiee[:, begin_x:fin_x, begin_y:fin_y]
                # patches[idx] = sous_image
                # idx += 1
                idx = i*nb_patches_y+j
                begin_x=i*(size_patches-size_overlay)
                begin_y=j*(size_patches-size_overlay)
                end_x=begin_x+size_patches
                end_y=begin_y+size_patches
                sous_image=image_modifiee[:, begin_x:end_x, begin_y:end_y]
                patches[idx] = sous_image    
                

        return patches, self.images_name[index], shape


def create_predict_generator(config):
    """Return the prediction's generator from a config """
    generator = PredictGenerator(config)
    return DataLoader(generator,
                      batch_size=1,
                      shuffle=False,
                      drop_last=False)