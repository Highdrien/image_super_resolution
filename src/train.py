import os
import numpy as np
from tqdm import tqdm

import torch

from src.model import get_model
from src.metrics import compute_metrics
from src.utils import save_learning_curves, find_best_from_csv
from src.dataloader import create_generator
from src.checkpoints import save_checkpoint_all, save_checkpoint_best, save_checkpoint_last, get_checkpoint_path

from config.utils import train_logger, train_step_logger

torch.manual_seed(0)


def train(config, resume_training=None):
    """
    makes a training according to the chosen configuration
    if resume_training == None -> do a new training
    else -> resume training from the path: "resume_training"
    """
    if resume_training is not None:
        print('Resume training from', resume_training)

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
    model.to(device)

    if resume_training is not None:
        # Load model's weight
        checkpoint_path = get_checkpoint_path(config, resume_training)
        print("checkpoint path:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        del checkpoint  # dereference

    # Loss
    if config.model.loss.lower() == 'mse':
        print('optimizer: mse')
        criterion = torch.nn.MSELoss()
    else:
        raise 'MSE loss is the only one to be implemented'

    # Optimizer
    if config.model.optimizer.lower() == "adam":
        print('optimizer: adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    elif config.model.optimizer.lower() == "sgd":
        print('optimizer: sgd')
        optimizer = torch.optim.SGD(model.parameters(), lr=config.model.learning_rate)
    else:
        raise 'please choose between adam or sgd optimizer'

    # Metrics
    metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))

    # Save training
    if resume_training is None:
        logging_path = train_logger(config)
        epoch_start = 1
        best_epoch, best_val_loss = 0, 10e6

    else:
        logging_path = resume_training
        best_epoch, best_val_loss, nb_epochs = find_best_from_csv(os.path.join(logging_path, 'train_log.csv'))
        epoch_start = nb_epochs + 1

    ###############################################################
    # Start Training                                              #
    ###############################################################
    model.train()

    for epoch in range(epoch_start, config.train.epochs + 1):
        print('epoch:' + str(epoch))
        train_loss = []
        train_metrics = np.zeros(len(metrics_name), dtype=float)

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
            train_metrics += compute_metrics(config, y_pred.detach(), y_true.detach())

            train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f" % (epoch, np.mean(train_loss)))
            train_range.refresh()
        
        train_loss = np.mean(train_loss)
        train_metrics = train_metrics / len(train_generator)

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        model.eval()
        val_loss = []
        val_metrics = np.zeros(len(metrics_name), dtype=float)
        val_range = tqdm(val_generator)

        with torch.no_grad():
            
            for (lr_image, hr_image) in val_range:
                lr_image = lr_image.to(device)
                y_true = hr_image.to(device)

                y_pred = model(lr_image)

                loss = criterion(y_pred, y_true)
                val_loss.append(loss.item())
                val_metrics += compute_metrics(config, y_true, y_pred)

                val_range.set_description("VAL   -> epoch: %4d || val_loss: %4.4f" % (epoch, np.mean(val_loss)))
                val_range.refresh()

        val_loss = np.mean(val_loss)
        val_metrics = val_metrics / len(val_generator)

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################

        train_step_logger(logging_path, epoch, train_loss, val_loss, train_metrics, val_metrics)

        if config.model.save_checkpoint == 'all':
            save_checkpoint_all(model, logging_path, epoch)

        elif config.model.save_checkpoint == 'best':
            best_epoch, best_val_loss = save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss)

    if config.model.save_checkpoint == 'best':
        save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss, end_training=True)

    elif config.model.save_checkpoint == 'last':
        save_checkpoint_last(config, model, logging_path)

    if config.train.save_learning_curves:
        save_learning_curves(logging_path)
