import rasterio
import numpy as np
from rasterio.enums import Resampling

image = "blended_img_cnt41.png"

# Open the large TIFF image
with rasterio.open(image) as src:
    scale_factor = 10  # Adjust as needed
    new_height = src.height // scale_factor
    new_width = src.width // scale_factor

    # Read and resample to smaller size
    preview = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.nearest  # Faster, use Resampling.bilinear for better quality
    )

    # Normalize for saving as an image
    preview = np.moveaxis(preview, 0, -1)  # Move bands to last dimension
    preview = (preview - preview.min()) / (preview.max() - preview.min()) * 255  # Normalize
    preview = preview.astype(np.uint8)  # Convert to 8-bit

    # Save as a smaller preview
    from PIL import Image
    Image.fromarray(preview).convert("RGB").save("compressed_quarry.png")
