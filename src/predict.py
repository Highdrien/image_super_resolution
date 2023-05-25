import os
from tqdm import tqdm
import numpy as np
import math

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
    size_overlay = 20
    upscale_facor = config.upscale_factor
    big_size_patches = size_patches * upscale_facor
    big_size_overlay = size_overlay * upscale_facor

    canal, x, y = image_shape

    # Calculate the number of patches in each dimension
    nb_patches_x = math.ceil((x-size_patches)/(size_patches-size_overlay))+1
    nb_patches_y = math.ceil((y-size_patches)/(size_patches-size_overlay))+1
    nb_patches = nb_patches_x * nb_patches_y
    # Calculate the modified image dimensions
    
    x_modifie = big_size_patches+(big_size_patches-big_size_overlay)*(nb_patches_x-1)
    y_modifie = big_size_patches+(big_size_patches-big_size_overlay)*(nb_patches_y-1)

    # Verified that the number of patches correspond to the number of images obtained
    assert nb_patches == prediction.shape[0], "The number of patches does not correspond to the size of the original image."

    # Create a tensor to stock the reconstructed image
    hr_image = torch.zeros((canal, x * upscale_facor, y * upscale_facor))
    # Create the tensor for the modified image with black border
    image_modifiee = torch.zeros((canal, x_modifie, y_modifie))

    # Reconstruct the image by assembling the patches
    # idx = 0
    for i in range(nb_patches_x):
        for j in range(nb_patches_y):
            idx = i*nb_patches_y+j
            begin_x=i*(big_size_patches-big_size_overlay)+(i>0)*big_size_overlay//2
            begin_y=j*(big_size_patches-big_size_overlay)+(j>0)*big_size_overlay//2
            end_x=i*(big_size_patches-big_size_overlay)+big_size_patches-(i<nb_patches_x-1)*big_size_overlay//2
            end_y=j*(big_size_patches-big_size_overlay)+big_size_patches-(j<nb_patches_y-1)*big_size_overlay//2
            centered_begin_x = i*(big_size_patches-big_size_overlay)+ big_size_overlay-(i>0)*big_size_overlay//2
            centered_begin_y = j*(big_size_patches-big_size_overlay)+ big_size_overlay-(j>0)*big_size_overlay//2
            centered_end_x = i*(big_size_patches-big_size_overlay)+big_size_patches - big_size_overlay + (i<nb_patches_x-1)*big_size_overlay//2
            centered_end_y = j*(big_size_patches-big_size_overlay)+big_size_patches - big_size_overlay + (j<nb_patches_y-1)*big_size_overlay//2

            begin_loc_x = (i>0)*big_size_overlay//2
            begin_loc_y = (j>0)*big_size_overlay//2
            end_loc_x = big_size_patches-(i<nb_patches_x-1)*big_size_overlay//2
            end_loc_y = big_size_patches-(j<nb_patches_y-1)*big_size_overlay//2
            center_begin_loc_x = big_size_overlay-(i>0)*big_size_overlay//2
            center_begin_loc_y = big_size_overlay-(j>0)*big_size_overlay//2
            center_end_loc_x = big_size_patches-big_size_overlay+(i<nb_patches_x-1)*big_size_overlay//2
            center_end_loc_y = big_size_patches-big_size_overlay+(j<nb_patches_y-1)*big_size_overlay//2

            # Center
            image_modifiee[:,centered_begin_x:centered_end_x,centered_begin_y:centered_end_y] = prediction[idx,:,center_begin_loc_x:center_end_loc_x,center_begin_loc_y:center_end_loc_y]
            # Upper left corner
            coeff = 0
            if not(i == 0):
                coeff = coeff + 2            
            if not(j == 0):
                coeff = coeff + 2
            if coeff == 0:
                coeff = 1
            image = torch.div(prediction[idx,:,begin_loc_x:center_begin_loc_x,begin_loc_y:center_begin_loc_y],coeff)
            image_modifiee[:,begin_x:centered_begin_x,begin_y:centered_begin_y] += image
            # Upper right corner
            coeff = 0
            if (not(i == nb_patches_x-1)):
                coeff = coeff + 2
            if not(j == 0):
                coeff = coeff + 2
            if coeff == 0:
                coeff = 1
            image = torch.div(prediction[idx,:,center_end_loc_x:end_loc_x,begin_loc_y:center_begin_loc_y],coeff)
            image_modifiee[:,centered_end_x:end_x,begin_y:centered_begin_y] += image
            # Lower left corner
            coeff = 0
            if not(i == 0):
                coeff = coeff + 2            
            if not(j == nb_patches_y-1):
                coeff = coeff + 2
            if coeff == 0:
                coeff = 1
                
            image = torch.div(prediction[idx,:,begin_loc_x:center_begin_loc_x,center_end_loc_y:end_loc_y],coeff)
            image_modifiee[:,begin_x:centered_begin_x,centered_end_y:end_y] += image
            # Lower right corner
            coeff = 0
            if not(i == nb_patches_x-1):
                coeff = coeff + 2            
            if not(j == nb_patches_y-1):
                coeff = coeff + 2
            if coeff == 0:
                coeff = 1
            image = torch.div(prediction[idx,:,center_end_loc_x:end_loc_x,center_end_loc_y:end_loc_y],coeff)
            image_modifiee[:,centered_end_x:end_x,centered_end_y:end_y] += image

            # Border right
            if not(i == 0):
                coeff =  2
            
            else:
                coeff = 1
            
            image = torch.div(prediction[idx,:,begin_loc_x:center_begin_loc_x, center_begin_loc_y:center_end_loc_y],coeff)
            image_modifiee[:,begin_x:centered_begin_x,centered_begin_y:centered_end_y] += image

            # Border left
            if not(i == nb_patches_x-1):
                coeff =  2
            else:
                coeff = 1

            image = torch.div(prediction[idx,:,center_end_loc_x:end_loc_x, center_begin_loc_y:center_end_loc_y],coeff)
            image_modifiee[:,centered_end_x:end_x,centered_begin_y:centered_end_y] += image
            
             # Upper border
            if not(j == 0):
                coeff = 2
            else:
                coeff = 1
            
            image = torch.div(prediction[idx,:, center_begin_loc_x:center_end_loc_x,begin_loc_y:center_begin_loc_y],coeff)
            image_modifiee[:,centered_begin_x:centered_end_x,begin_y:centered_begin_y] += image

            # Lower border
            if not(j == nb_patches_y-1):
                coeff =  2
            else:
                coeff = 1
            image = torch.div(prediction[idx,:, center_begin_loc_x:center_end_loc_x,center_end_loc_y:end_loc_y],coeff)
            image_modifiee[:,centered_begin_x:centered_end_x,centered_end_y:end_y] += image

            # begin_x = i * big_size_patches
            # begin_y = j * big_size_patches
            # end_x = begin_x + big_size_patches
            # end_y = begin_y + big_size_patches

            # if end_x > x * upscale_facor:
            #     end_x = x * upscale_facor
            # if end_y > y * upscale_facor:
            #     end_y = y * upscale_facor

            # end_patches_x = int(end_x - begin_x)
            # end_patches_y =  int(end_y - begin_y)

            # hr_image[:, begin_x:end_x, begin_y:end_y] = prediction[idx, :, :end_patches_x, :end_patches_y]
            # idx += 1
    hr_image=image_modifiee[:,:x*upscale_facor,:y*upscale_facor]

    saving_path = os.path.join(dst_path, image_name[:-4] + config.predict.end_image_name + image_name[-4:])

    image = transforms.ToPILImage()(hr_image)
    image.save(saving_path)