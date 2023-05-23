import os
from tqdm import tqdm

import torch

from src.model import get_model
from src.dataloader import create_generator

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
    checkpoint_path = get_checkpoint_path(config, logging_path)
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

    ###############################################################
    # Start Evaluation                                            #
    ###############################################################
    model.eval()

    test_loss = 0
    # train_metrics = np.zeros(len(metrics_name), dtype=float)

    for (lr_image, hr_image) in tqdm(test_generator):
        lr_image = lr_image.to(device)
        y_true = hr_image.to(device)

        y_pred = model(lr_image)

        loss = criterion(y_pred, y_true)
        test_loss += loss.item()
        
    test_loss = test_loss / len(test_generator)
    print('test loss:', test_loss)

    test_logger(logging_path, [config.model.loss], [test_loss])



def get_checkpoint_path(config, path):
    pth_in_path = list(filter(lambda x: x[-3:] == 'pth', os.listdir(path)))

    if len(pth_in_path) == 1:
        return os.path.join(path, pth_in_path[0])

    if len(pth_in_path) == 0 and 'checkpoint_path' in os.listdir(path):
        model_path = os.path.join(path, 'checkpoint_path')
        print(model_path)

        if config.test.checkpoint in os.listdir(model_path):
            return os.path.join(model_path, config.test.checkpoint)

        elif config.test.checkpoint == 'last':
            pth_in_checkpoint = list(filter(lambda x: x[-3:] == 'pth', os.listdir(model_path)))
            model_name = 'model' + str(len(pth_in_checkpoint)) + 'pth'
            return os.path.join(model_path, model_name)

        elif 'model' + config.test.checkpoint + 'pth' in os.listdir(model_path):
            return os.path.join(model_path, 'model' + config.test.checkpoint + 'pth')

    elif config.test.checkpoint == 'last':
        return os.path.join(path, pth_in_path[-1])

    elif 'model' + config.test.checkpoint + 'pth' in os.listdir(path):
        return os.path.join(path, 'model' + config.test.checkpoint + 'pth')

    raise 'The model weights could not be found'