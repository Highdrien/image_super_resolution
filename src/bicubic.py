import os
from tqdm import tqdm

import torch
from torch import nn

from src.model import get_model
from src.dataloader import create_generator

from config.utils import test_logger

torch.manual_seed(0)


def bicubic(lr_image, up_scale_factor):
    upsample = nn.Upsample(scale_factor=up_scale_factor, mode='nearest')
    hr_image = upsample(lr_image)
    return hr_image



def test_bicubic(config):
    test_generator = create_generator(config, mode='test')
    logging_path = os.path.join("logs", "bicubic")

    # Loss
    if config.model.loss.lower() == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise 'MSE loss is the only one to be implemented'

    ###############################################################
    # Start Evaluation                                            #
    ###############################################################

    test_loss = 0
    # train_metrics = np.zeros(len(metrics_name), dtype=float)

    for (lr_image, y_true) in tqdm(test_generator):

        y_pred = bicubic(lr_image, config.upscale_factor)

        loss = criterion(y_pred, y_true)
        test_loss += loss.item()
        
    test_loss = test_loss / len(test_generator)
    print('test loss:', test_loss)

    if not(os.path.exists(logging_path)):
        os.mkdir(logging_path)
    test_logger(logging_path, [config.model.loss], [test_loss], config.upscale_factor)

