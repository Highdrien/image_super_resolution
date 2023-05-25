import os
from tqdm import tqdm
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

from src.model import get_model
from src.checkpoints import get_checkpoint_path
from src.dataloader import create_predict_generator


def predict(logging_path, config):
    """
    makes a test of an already trained model and creates a test_log.csv file in the experiment_path containing the
    metrics values at the end of the test
    :param logging_path: path of the experiment folder, containing the config, and the model weights
    :param config: configuration of the model
    """
    predict_generator = create_predict_generator(config)

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

    ###############################################################
    # Start Prediction                                            #
    ###############################################################
    model.eval()

    for patches, image_name, image_shape in tqdm(predict_generator):
        patches = patches.to(device)

        prediction = model(patches.squeeze())

        save_image(config, prediction, image_name[0], image_shape)
    



def save_image(config, prediction, image_name, image_shape):
    """
    save the prediction in the dst_path
    """
    dst_path = config.predict.dst_path

    if not(os.path.exists(dst_path)):
        os.mkdir(dst_path)

    size_patches = config.data.image_size
    upscale_facor = config.upscale_factor
    big_size_patches = size_patches * upscale_facor

    canal, x, y = image_shape
    x_modified = x + (size_patches - x % size_patches) if x % size_patches != 0 else x
    y_modified = y + (size_patches - y % size_patches) if y % size_patches != 0 else y    

    # Calculer le nombre de sous-images dans chaque dimension
    nb_patches_x = x_modified // size_patches
    nb_patches_y = y_modified // size_patches
    nb_patches = nb_patches_x * nb_patches_y


    # Vérifier que le nombre de sous-images correspond
    assert nb_patches == prediction.shape[0], "The number of patches does not correspond to the size of the original image."

    # Créer un tenseur pour stocker l'image reconstituée
    hr_image = torch.zeros((canal, x * upscale_facor, y * upscale_facor))

    # Reconstituer l'image en assemblant les sous-images
    idx = 0
    for i in range(nb_patches_x):
        for j in range(nb_patches_y):
            begin_x = i * big_size_patches
            begin_y = j * big_size_patches
            end_x = begin_x + big_size_patches
            end_y = begin_y + big_size_patches

            if end_x > x * upscale_facor:
                end_x = x * upscale_facor
            if end_y > y * upscale_facor:
                end_y = y * upscale_facor

            end_patches_x = int(end_x - begin_x)
            end_patches_y =  int(end_y - begin_y)

            hr_image[:, begin_x:end_x, begin_y:end_y] = prediction[idx, :, :end_patches_x, :end_patches_y]
            idx += 1
    
    saving_path = os.path.join(dst_path, image_name[:-4] + config.predict.end_image_name + image_name[-4:])

    image = transforms.ToPILImage()(hr_image)
    image.save(saving_path)