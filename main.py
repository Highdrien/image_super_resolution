import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.dataloader import create_generator
from src.train import train


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def main(options):
    if options['mode'] == 'data':
        config = load_config(options['config_path'])
        create_generator(config, 'train')

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        train(config)
    
    else:
        print('ERROR')
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'data', 'train', 'test' and 'predict'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config")

    args = parser.parse_args()
    options = vars(args)

    main(options)