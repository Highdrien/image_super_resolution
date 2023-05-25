import os
import torch


def save_checkpoint_all(model, logging_path, epoch):
    """
    save all training checkpoints
    """
    checkpoint_path = os.path.join(logging_path, 'checkpoint_path')
    checkpoint_name = 'model' + str(epoch) + '.pth'
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))


def save_checkpoint_best(model, logging_path, epoch, best_epoch, val_loss, best_val_loss, end_training=False):
    """
    save the best training checkpoint acoording the validation's loss
    """
    if not(end_training):
        # at the end of one epoch
        if val_loss < best_val_loss:
            print('saving checkpoints')
            best_epoch, best_val_loss = epoch, val_loss
            torch.save(model.state_dict(), os.path.join(logging_path, 'model.pth'))
        
    else:
        # at the end of og the training
        old_name = os.path.join(logging_path, 'model.pth')
        new_name = os.path.join(logging_path, 'model' + str(best_epoch) + '.pth')
        os.rename(old_name, new_name)
    
    return best_epoch, best_val_loss


def save_checkpoint_last(config, model, logging_path):
    """
    save the last training checkpoint 
    """
    torch.save(model.state_dict(), os.path.join(logging_path, 'model' + str(config.train.epochs) + '.pth'))


def get_checkpoint_path(config, path, mode):
    """
    get a checkpoint file according the config and the experiment_path
    use for test and prediction
    """
    pth_in_path = list(filter(lambda x: x[-3:] == 'pth', os.listdir(path)))

    if len(pth_in_path) == 1:
        return os.path.join(path, pth_in_path[0])

    if len(pth_in_path) == 0 and 'checkpoint_path' in os.listdir(path):
        model_path = os.path.join(path, 'checkpoint_path')
        print(model_path)

        if config.test.checkpoint in os.listdir(model_path):
            return os.path.join(model_path, config.test.checkpoint)

        elif config.test.checkpoint == 'last' or config.test.checkpoint == 'all':
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