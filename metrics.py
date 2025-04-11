from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# Load aligned images
# stitched = cv2.imread("./src/golf_source_30_aligned.png", cv2.IMREAD_GRAYSCALE)
stitched = cv2.imread("./src/highway_source_56_aligned_cropped.png", cv2.IMREAD_GRAYSCALE)
# aligned_reference = cv2.imread("./src/golf_reference_30_aligned.png", cv2.IMREAD_GRAYSCALE)
aligned_reference = cv2.imread("./src/highway_reference_56_aligned_cropped.png", cv2.IMREAD_GRAYSCALE)

# Check if images are loaded properly
if stitched is None or aligned_reference is None:
    print("Error: Could not load one or both images.")
else:
    # Compute SSIM
    ssim_score, _ = ssim(stitched, aligned_reference, full=True)
    print(f"SSIM Score: {ssim_score:.4f}")

    # Compute MSE
    mse_value = np.mean((stitched.astype("float") - aligned_reference.astype("float")) ** 2)
    print(f"MSE: {mse_value:.4f}")

    # Compute PSNR
    if mse_value == 0:
        psnr_score = 100  # Perfect match
    else:
        psnr_score = 10 * np.log10(255**2 / mse_value)

    print(f"PSNR Score: {psnr_score:.2f} dB")
