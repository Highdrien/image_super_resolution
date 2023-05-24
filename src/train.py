import os
import numpy as np
from tqdm import tqdm

import torch
from torchmetrics import PeakSignalNoiseRatio

from src.model import get_model
from src.utils import save_learning_curves
from src.dataloader import create_generator

from config.utils import train_logger, train_step_logger

torch.manual_seed(0)


def train(config):
    """
    makes a training according to the chosen configuration
    """
    train_generator = create_generator(config, mode='train')
    val_generator = create_generator(config, mode='val')


    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # Model
    model = get_model(config)
    # model.to(torch.float)
    model.to(device)

    # Loss
    if config.model.loss.lower() == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise 'MSE loss is the only one to be implemented'

    # Optimizer
    if config.model.optimizer.lower() == "adam":
        print('loss: adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    elif config.model.optimizer.lower() == "sgd":
        print('loss: sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=config.model.learning_rate)
    else:
        raise 'please choose between adam or sgd optimizer'

    # Save training
    logging_path = train_logger(config)

    best_epoch, best_val_loss = 0, 10e6

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    for epoch in range(1, config.train.epochs + 1):
        print('epoch:' + str(epoch))
        train_loss = []
        train_psnr = []
        # train_metrics = np.zeros(len(metrics_name), dtype=float)

        train_range = tqdm(train_generator)
        for (lr_image, hr_image) in train_range:
            lr_image = lr_image.to(device)
            y_true = hr_image.to(device)

            y_pred = model(lr_image)

            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            psnr_tensor = psnr(y_pred, y_true)
            train_psnr.append(psnr_tensor.item())

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()
        
        train_loss = np.mean(train_loss)
        train_psnr = np.mean(train_psnr)

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        model.eval()
        val_loss = []
        val_psnr = []
        val_range = tqdm(val_generator)

        with torch.no_grad():
            
            for (lr_image, hr_image) in val_range:
                lr_image = lr_image.to(device)
                y_true = hr_image.to(device)

                y_pred = model(lr_image)

                loss = criterion(y_pred, y_true)
                val_loss.append(loss.item())
                psnr_tensor = psnr(y_pred, y_true)
                val_psnr.append(psnr_tensor.item())


                val_range.set_description("VAL   -> epoch: %4d || val_loss: %4.4f" % (epoch, np.mean(val_loss)))
                val_range.refresh()
                # val_metrics += compute_metrics(config, y_true, y_pred, argmax_axis=-1)

        val_loss = np.mean(val_loss)
        val_psnr = np.mean(val_psnr)

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################

        train_step_logger(logging_path, epoch, train_loss, val_loss, train_psnr, val_psnr)

        if config.train.save_checkpoint.lower() == 'all':
            checkpoint_path = os.path.join(logging_path, 'checkpoint_path')
            checkpoint_name = 'model' + str(epoch) + 'pth'
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))

        elif config.train.save_checkpoint.lower() == 'best':
            if val_loss < best_val_loss:
                print('saving checkpoints')
                best_epoch, best_val_loss = epoch, val_loss
                torch.save(model.state_dict(), os.path.join(logging_path, 'model.pth'))

    if config.train.save_checkpoint.lower() == 'best':
        old_name = os.path.join(logging_path, 'model.pth')
        new_name = os.path.join(logging_path, 'model' + str(best_epoch) + '.pth')
        os.rename(old_name, new_name)

    elif config.train.save_checkpoint.lower() == 'last':
        torch.save(model.state_dict(), os.path.join(logging_path, 'model' + str(config.train.epochs + 1) + '.pth'))

    if config.train.save_learning_curves:
        save_learning_curves(logging_path)
