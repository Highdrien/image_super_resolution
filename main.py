import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.dataloader import getbatch
from src.train import train
from src.test import test
from src.bicubic import test_bicubic


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
    if options['mode'] == 'data':
        config = load_config(options['config_path'])
        getbatch(config, 'train')

    elif options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    
    elif options['mode'] == 'test':
        config_path = os.path.join(options['path'], find_config(options['path']))
        config = load_config(config_path)
        test(options['path'], config)

    elif options['mode'] == 'bicubic':
        config = load_config(options['config_path'])
        test_bicubic(config)

    else:
        print(options['mode'])
        print('ERROR: please chose between data, train, test')
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'data', 'train', 'test' and 'predict'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config")
    parser.add_argument('--path', type=str, help="experiment path (only for test or prediction)")

    args = parser.parse_args()
    options = vars(args)

    main(options)