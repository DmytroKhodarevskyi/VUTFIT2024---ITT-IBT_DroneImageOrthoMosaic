import os
from PIL import Image

# Paths
src_folder = os.path.join("src", "DroneMapper_Golf9_May2016")
output_folder = os.path.join("src", "resized_photos")

# Target resolution
target_width = 1920
target_height = 1080

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def resize_image(input_path, output_path, target_width, target_height):
    """Resize an image to the target resolution."""
    with Image.open(input_path) as img:
        # Preserve aspect ratio
        img.thumbnail((target_width, target_height))
        img.save(output_path)
        print(f"Resized and saved: {output_path}")

# Process each image in the src folder
for filename in os.listdir(src_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Include common image formats
        input_path = os.path.join(src_folder, filename)
        output_path = os.path.join(output_folder, filename)
        resize_image(input_path, output_path, target_width, target_height)

print(f"All images resized and saved in {output_folder}.")
