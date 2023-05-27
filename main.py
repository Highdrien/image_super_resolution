import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test
from src.predict import predict
from src.bicubic import test_bicubic, predict_bicubic
from src.tranfer_learning import tranfer_learning


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def find_config(experiment_path):
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))
    
    if len(yaml_in_path) == 1:
        return yaml_in_path[0]

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .ymal was found in", experiment_path)
    
    exit()


def main(options):
    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    
    elif options['mode'] == 'test':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        test(options['path'], config)

    elif options['mode'] == 'predict':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        predict(options['path'], config)
    
    elif options['mode'] == 'resumetrain':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        train(config, resume_training=options['path'])

    elif options['mode'] == 'bicubic':
        config = load_config(options['config_path'])
        test_bicubic(config)
    
    elif options['mode'] in ['tf', 'tranfer_learning']:
        previous_config_path = os.path.join(options['path'], find_config(options['path']))
        previous_config = load_config(previous_config_path)
        tranfer_learning(previous_config, options['path'], options['new_upscale_factor'])

    elif options['mode'] == 'predict_bicubic':
        config = load_config(options['config_path'])
        predict_bicubic(config)

    else:
        print(options['mode'])
        print('ERROR: please chose between data, train, test, bicubic, predict or resumetrain')
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test', 'predict', 'bicubic' and 'resumetrain'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or resume previous training)")
    parser.add_argument('--new_upscale_factor', type=int, help="new upscale factor for a tranfere learning")

    args = parser.parse_args()
    options = vars(args)

    main(options)