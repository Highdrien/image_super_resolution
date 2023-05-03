import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def center_patches(src_folder, image_name, patch_size, dst_folder):
    """
    take one image from src_folder and save the centre patches in dst_folder
    """
    image = np.asarray(Image.open(os.path.join(src_folder, image_name)))

    n, m, _ = np.shape(image)
    w, h = patch_size
    centre_x, centre_y = n // 2, m // 2
            
    patch = image[centre_x - w // 2 : centre_x + w // 2, centre_y - h // 2 : centre_y + h // 2, :]

    if patch.shape != (w, h, 3):
        print('ERROR patch shape for', os.path.join(src_folder, image_name))
        exit()

    img = Image.fromarray(patch)
    img.save(os.path.join(dst_folder, image_name))


def save_patches(src_folder, patch_size):
    """
    take all the image from the src_folder to find the centre patches on it and save it on the patches/src_folder
    """

    dst_folder = os.path.join('patches', src_folder)
    src_folder = os.path.join('DIV2K', src_folder)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    for image_name in tqdm(os.listdir(src_folder), desc='find and save patches from ' + src_folder):
        if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg'):
            center_patches(src_folder, image_name, patch_size, dst_folder)



def main():
    patch_size = (120, 120)              # chose 120 because it's divide by 2, 3 and 4
    patch_size = np.array(patch_size)

    # PATH
    train_HR = 'DIV2K_train_HR'
    train_X2 = os.path.join('DIV2K_train_LR_bicubic', 'X2')
    train_X3 = os.path.join('DIV2K_train_LR_bicubic', 'X3')
    train_X4 = os.path.join('DIV2K_train_LR_bicubic', 'X4')

    valid_HR = os.path.join('DIV2K_valid_HR')
    valid_X2 = os.path.join('DIV2K_valid_LR_bicubic', 'X2')
    valid_X3 = os.path.join('DIV2K_valid_LR_bicubic', 'X3')
    valid_X4 = os.path.join('DIV2K_valid_LR_bicubic', 'X4')


    save_patches(train_HR, patch_size)
    save_patches(train_X2, patch_size // 2)
    save_patches(train_X3, patch_size // 3)
    save_patches(train_X4, patch_size // 4)

    save_patches(valid_HR, patch_size)
    save_patches(valid_X2, patch_size // 2)
    save_patches(valid_X3, patch_size // 3)
    save_patches(valid_X4, patch_size // 4)


 
if __name__ == '__main__':
    main()