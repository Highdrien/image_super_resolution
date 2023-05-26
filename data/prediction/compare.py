import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def compare_images(image1, image2, image3, zoom_factor, upscale_factor, experiment, saving_path):
    # Charger les images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img3 = cv2.imread(image3)

    # Redimensionner la première image
    height, width, _ = img1.shape
    img1_resized = cv2.resize(img1, (width*upscale_factor, height*upscale_factor))

    # Vérifier que les images ont la même taille
    if img1_resized.shape != img2.shape or img1_resized.shape != img3.shape:
        print("Les images n'ont pas la même taille.")
        return

    # Afficher les images dans la même fenêtre
    fig, axes = plt.subplots(2, 3)

    # Afficher l'image 1 entière (plus petite)
    axes[0, 0].imshow(cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Low resolution image')

    # Afficher l'image 2 entière
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Bicubic')

    # Afficher l'image 3 entière
    axes[0, 2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Experiment ' + str(experiment))

    # Calculer le zoom
    zoom_height = int(height * zoom_factor)
    zoom_width = int(width * zoom_factor)

    zoom_height_img1 = int(height * zoom_factor / upscale_factor)
    zoom_width_img1 = int(width * zoom_factor / upscale_factor)

    # Centrer le zoom
    start_row = int(upscale_factor * height/2 - zoom_height/2)
    end_row = int(upscale_factor * height/2 + zoom_height/2)
    start_col = int(upscale_factor * width/2 - zoom_width/2)
    end_col = int(upscale_factor * width/2 + zoom_width/2)

    # Extraire les régions zoomées des images
    zoomed_img1 = img1[int(height/2 - zoom_height_img1/2):int(height/2 + zoom_height_img1/2), 
                       int(width/2 - zoom_width_img1/2):int(width/2 + zoom_width_img1/2)]
    zoomed_img2 = img2[start_row:end_row, start_col:end_col]
    zoomed_img3 = img3[start_row:end_row, start_col:end_col]

    zoom = height / zoom_height_img1

    # Afficher le zoom de l'image 1
    axes[1, 0].imshow(cv2.cvtColor(zoomed_img1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Zoom x{0:.2f}".format(zoom))

    # Afficher le zoom de l'image 2
    axes[1, 1].imshow(cv2.cvtColor(zoomed_img2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Zoom x{0:.2f}".format(zoom))

    # Afficher le zoom de l'image 3
    axes[1, 2].imshow(cv2.cvtColor(zoomed_img3, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("Zoom x{0:.2f}".format(zoom))

    # Masquer les axes
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(saving_path)

def main(zoom_factor):

    experiments = [{"number": 1, "upscale_factor": 3}, 
                   {"number": 2, "upscale_factor": 3}, 
                   {"number": 3, "upscale_factor": 2},
                   {"number": 4, "upscale_factor": 4}]
    
    images_name = [("person_rgb", ".png"), ("lac", ".jpg")]

    for experiment in tqdm(experiments):

        for name, end in images_name:
            original_image = os.path.join("src", name + end)
            bicubic_image = os.path.join("dst", name + "_bicubic" + str(experiment["upscale_factor"]) + end)
            predicted_image = os.path.join("dst", name +"_experiment" + str(experiment["number"]) + end)
            saving_path = os.path.join("comparaison", name + "_exp" + str(experiment["number"]) + ".png")
            compare_images(original_image, 
                            bicubic_image, 
                            predicted_image,
                            zoom_factor=zoom_factor, 
                            upscale_factor=experiment["upscale_factor"], 
                            experiment=experiment["number"],
                            saving_path=saving_path)

main(zoom_factor=0.3)
        

        
