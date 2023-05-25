import os
import numpy as np
from tqdm import tqdm

import torch

from src.model import get_model
from src.metrics import compute_metrics
from src.dataloader import create_generator
from src.checkpoints import get_checkpoint_path

from config.utils import test_logger

torch.manual_seed(0)


def test(logging_path, config):
    """
    makes a test of an already trained model and creates a test_log.csv file in the experiment_path containing the
    metrics values at the end of the test
    :param logging_path: path of the experiment folder, containing the config, and the model weights
    :param config: configuration of the model
    """
    test_generator = create_generator(config, mode='train')

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Model
    model = get_model(config)
    model.to(device)

    # Load model's weight
    checkpoint_path = get_checkpoint_path(config, logging_path, mode='test')
    print("checkpoint path:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    del checkpoint  # dereference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    model.eval()

    test_loss = 0
    test_metrics = np.zeros(len(metrics_name), dtype=float) #TODO: add metrics in the test

    for (lr_image, hr_image) in tqdm(test_generator):
        lr_image = lr_image.to(device)
        y_true = hr_image.to(device)

        y_pred = model(lr_image)

        loss = criterion(y_pred, y_true)
        test_loss += loss.item()
        test_metrics += compute_metrics(config, y_pred.detach(), y_true.detach())
        
    test_loss = test_loss / len(test_generator)
    print('test loss:', test_loss)

    metrics_name = [config.model.loss] + metrics_name
    metrics_value = [test_loss] + list(test_metrics)

    test_logger(logging_path, metrics_name, metrics_value, config.upscale_factor)
