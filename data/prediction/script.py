from PIL import Image

# Ouvrir l'image
image_rgba = Image.open("src\\person.png")

# Convertir l'image RGBA en RGB
image_rgb = image_rgba.convert("RGB")

# Vérifier la nouvelle forme de l'image
image_rgb.save("src\\person_rgb.png")
