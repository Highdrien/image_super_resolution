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


def main(options):
    config = load_config(options['config_path'])

    if options['mode'] == 'data':
        getbatch(config, 'train')

    elif options['mode'] == 'train':
        train(config)
    
    elif options['mode'] == 'test':
        test(options['path'], config)

    elif options['mode'] == 'bicubic':
        test_bicubic(config)

    else:
        print(options['mode'])
        print('ERROR')
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'data', 'train', 'test' and 'predict'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config")
    parser.add_argument('--path', type=str, help="experiment path")

    args = parser.parse_args()
    options = vars(args)

    main(options)