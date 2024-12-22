import ImageStitchingData
import cv2 
import numpy as np

class Stitcher:
    def __init__(self, stitching_data):
        """
        Initializes a Stitcher object.

        :param stitching_data: An ImageStitchingData object containing images and their transformation matrices.
        """
        self.stitching_data = stitching_data

    def gradient_blend(self, image1, image2, mask1, mask2):
        """
        Blend two images using gradient masks for smooth transitions.
        
        :param image1: First image (numpy array).
        :param image2: Second image (numpy array).
        :param mask1: Gradient mask for the first image.
        :param mask2: Gradient mask for the second image.
        :return: A blended image.
        """
        # Ensure masks are normalized to the range [0, 1]
        if mask1.max() > 1: mask1 = mask1.astype(np.float32) / 255.0
        if mask2.max() > 1: mask2 = mask2.astype(np.float32) / 255.0

        # Ensure masks have the same spatial dimensions as the images
        if len(image1.shape) == 3 and len(mask1.shape) == 2:
            mask1 = np.repeat(mask1[:, :, np.newaxis], image1.shape[2], axis=2)
        if len(image2.shape) == 3 and len(mask2.shape) == 2:
            mask2 = np.repeat(mask2[:, :, np.newaxis], image2.shape[2], axis=2)

        # Combine masks to ensure they sum to 1 in overlapping areas
        combined_mask = mask1 + mask2
        combined_mask[combined_mask == 0] = 1  # Avoid division by zero

        # Blend the images using the masks
        blended_image = (image1.astype(np.float32) * mask1 +
                        image2.astype(np.float32) * mask2) / combined_mask

        return blended_image.astype(np.uint8)

    def create_gradient_mask(self, image):
        """
        Create a gradient mask for blending that increases weights towards the center of the image content.
        
        :param image: Input image (numpy array). Non-zero areas define the mask region.
        :return: A gradient mask (float32) with values between 0 and 1.
        """
        # Create a binary mask (non-zero pixels are set to 1)
        binary_mask = (image > 0).astype(np.uint8)

        if binary_mask.ndim == 3:  # If the image has multiple channels
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

        # Compute the distance transform
        distance_transform = cv2.distanceTransform(binary_mask, distanceType=cv2.DIST_L2, maskSize=5)

        # Normalize to the range [0, 1]
        gradient_mask = cv2.normalize(distance_transform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return gradient_mask

    def stitch_images(self, blending=False, gradient=False):
        """
        Warps all images by their corresponding homography matrices and combines them into one final image.

        :return: A single stitched image.
        """
        # Find the size of the final image by applying transformations and determining bounds
        all_images = self.stitching_data.get_all_images()
        all_homographies = self.stitching_data.get_all_homography_matrices()

        # Calculate bounds for the output canvas
        output_corners = []
        for img, H in zip(all_images, all_homographies):
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1],
                [w, h, 1],
                [0, h, 1]
            ]).T
            transformed_corners = H @ corners
            transformed_corners /= transformed_corners[2]  # Normalize
            output_corners.append(transformed_corners[:2].T)

        # Determine output canvas size
        all_corners = np.vstack(output_corners)
        x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)

        canvas_width = x_max - x_min
        canvas_height = y_max - y_min

        # Create an output canvas
        final_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Offset to align images on the canvas
        offset = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        # Warp each image and blend it into the final canvas
        for img, H in zip(all_images, all_homographies):
            transformed_H = offset @ H
            warped_image = cv2.warpPerspective(img, transformed_H, (canvas_width, canvas_height))
            
            # mask = (warped_image > 0).astype(np.uint8)

            if blending:
                # Create masks for the additional and main images
                mask_additional = (warped_image > 0).astype(np.float32)
                # mask_main = 1 - mask_additional
                mask_main = (final_image > 0).astype(np.float32)

                # Calculate combined weights for blending
                combined_mask = mask_additional + mask_main
                combined_mask[combined_mask == 0] = 1  # Avoid division by zero

                final_image = ((final_image.astype(np.float32) + warped_image.astype(np.float32)) / combined_mask).astype(np.uint8)


                # Perform 50-50 blending on overlapping regions
                # final_image = ((final_image.astype(np.float32) * 0.5 +
                                # warped_image.astype(np.float32) * 0.5) / combined_mask).astype(np.uint8)
                # final_image = (final_image * mask_main) + (warped_image * mask_additional)
            elif gradient:
                # mask_additional = (warped_image > 0).astype(np.uint8)
                mask_additional = self.create_gradient_mask(warped_image)
                # mask_main = (final_image > 0).astype(np.uint8)
                mask_main = self.create_gradient_mask(final_image)

                # final_image = self.feather_blend(final_image, warped_image, mask_main, mask_additional)
                final_image = self.gradient_blend(final_image, warped_image, mask_main, mask_additional)
            else:
                mask_additional = (warped_image > 0).astype(np.uint8)
                # mask_main = (final_image > 0).astype(np.uint8)
                mask_main = 1 - mask_additional

                final_image = (final_image * mask_main) + (warped_image * mask_additional)
            # mask = cv2.cvtColor((warped_image > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)

            # final_image = cv2.add(final_image, warped_image, mask=mask)

        return final_image

   