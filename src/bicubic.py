import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn

from src.dataloader import create_generator
from src.metrics import compute_metrics

from config.utils import test_logger

torch.manual_seed(0)


def bicubic(lr_image, up_scale_factor):
    upsample = nn.Upsample(scale_factor=up_scale_factor, mode='nearest')
    hr_image = upsample(lr_image)
    return hr_image



def test_bicubic(config):

    for i in range(2, 5):
        config.upscale_factor = i
        print("upscale factor:", i)

        test_generator = create_generator(config, mode='test')
        logging_path = os.path.join("logs", "bicubic")

        # Loss
        if config.model.loss.lower() == 'mse':
            criterion = torch.nn.MSELoss()
        else:
            raise 'MSE loss is the only one to be implemented'
        
        # Metrics
        metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))

        ###############################################################
        # Start Evaluation                                            #
        ###############################################################

        test_loss = 0
        test_metrics = np.zeros(len(metrics_name), dtype=float)

        for (lr_image, y_true) in tqdm(test_generator):

            y_pred = bicubic(lr_image, config.upscale_factor)

            loss = criterion(y_pred, y_true)
            test_loss += loss.item()
            test_metrics += compute_metrics(config, y_pred.detach(), y_true.detach())
            
        test_loss = test_loss / len(test_generator)
        test_metrics = test_metrics / len(test_generator)
        print('test loss:', test_loss)

        if not(os.path.exists(logging_path)):
            os.mkdir(logging_path)

        metrics_name = [config.model.loss] + metrics_name
        metrics_value = [test_loss] + list(test_metrics)

        test_logger(logging_path, metrics_name, metrics_value, config.upscale_factor)

