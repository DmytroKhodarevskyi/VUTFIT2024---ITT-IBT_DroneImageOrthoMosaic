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

    # def gradient_blend_along_edge(self, image1, image2, mask1, mask2, gradient_width):
    #     """
    #     Create a gradient blend along the edge of the overlapping region of two images,
    #     while keeping non-overlapping regions at full intensity.

    #     :param image1: First image (numpy array).
    #     :param image2: Second image (numpy array).
    #     :param mask1: Binary mask for the first image (1 where image1 is valid, 0 otherwise).
    #     :param mask2: Binary mask for the second image (1 where image2 is valid, 0 otherwise).
    #     :param gradient_width: Width of the gradient blend along the overlap edge (in pixels).
    #     :return: A blended image.
    #     """
    #     import cv2
    #     import numpy as np


    #     if mask1.ndim == 3:  # If the image has multiple channels
    #         mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
    #     if mask2.ndim == 3:  # If the image has multiple channels
    #         mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

    #     if mask1.all() == 0:
    #         return image2

    #     cv2.imwrite("mask1.png", mask1 * 255)
    #     cv2.imwrite("mask2.png", mask2 * 255)
    #     # Compute the overlap region
    #     overlap_mask = mask1 & mask2

    #     print(overlap_mask)

    #     cv2.imwrite("overlap_mask.png", overlap_mask)

    #     im1mask = overlap_mask + mask1

    #     # Dilate the overlap edges by the specified gradient width
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gradient_width, gradient_width))
    #     dilated_mask1 = cv2.dilate(mask1, kernel)
    #     # dilated_mask2 = cv2.dilate(mask2, kernel)

    #     cv2.imwrite("dilated_mask1.png", dilated_mask1 * 255)

    #     # Restrict the gradient region to the dilated overlap area
    #     gradient_region1 = (dilated_mask1 & ~mask1).astype(np.uint8)
    #     # gradient_region2 = (dilated_mask2 & ~mask2).astype(np.uint8)

    #     cv2.imwrite("gradient_region1.png", gradient_region1 * 255)

    #     # # Generate distance transforms within the gradient regions
    #     distance_transform1 = cv2.distanceTransform(gradient_region1, cv2.DIST_L2, maskSize=5)
    #     # distance_transform2 = cv2.distanceTransform(gradient_region2, cv2.DIST_L2, maskSize=5)

    #     # # Normalize the distance transforms to [0, 1] within the gradient width
    #     weight1 = 1 - np.clip(distance_transform1 / gradient_width, 0, 1)
    #     # weight2 = 1 - np.clip(distance_transform2 / gradient_width, 0, 1)

    #     weight1 *= mask2


    #     cv2.imwrite("weight1.png", weight1 * 255)

    #     im2mask = mask2 - overlap_mask - weight1
    #     im1mask = im1mask + weight1

    #     # blended_image = (image1 * im1mask + image2 * im2mask)
    #     for i in range(3):
    #         image1[:, :, i] = image1[:, :, i] * im1mask
    #         image2[:, :, i] = image2[:, :, i] * im2mask

    #     blended_image = image1 + image2

    #     # # Combine weights only within the overlap and gradient regions
    #     # combined_weights = weight1 + weight2
    #     # combined_weights[combined_weights == 0] = 1  # Avoid division by zero
    #     # normalized_weight1 = weight1 / combined_weights
    #     # normalized_weight2 = weight2 / combined_weights

    #     # # Apply full intensity outside the gradient and overlap regions
    #     # normalized_weight1[mask1 & ~overlap_mask] = 1
    #     # normalized_weight2[mask2 & ~overlap_mask] = 1

    #     # # Ensure weights match the dimensions of the images
    #     # if len(image1.shape) == 3 and normalized_weight1.ndim == 2:
    #     #     normalized_weight1 = np.repeat(normalized_weight1[:, :, np.newaxis], image1.shape[2], axis=2)
    #     # if len(image2.shape) == 3 and normalized_weight2.ndim == 2:
    #     #     normalized_weight2 = np.repeat(normalized_weight2[:, :, np.newaxis], image2.shape[2], axis=2)

    #     # # Blend the images using the weights
    #     # blended_image = (image1.astype(np.float32) * normalized_weight1 +
    #     #                 image2.astype(np.float32) * normalized_weight2)

    #     return blended_image.astype(np.uint8)


    def gradient_blend_along_edge(self, image1, image2, mask1, mask2, gradient_width):
        """
        Create a gradient blend along the edge of the overlapping region of two images,
        while keeping non-overlapping regions at full intensity.

        :param image1: First image (numpy array).
        :param image2: Second image (numpy array).
        :param mask1: Binary mask for the first image (1 where image1 is valid, 0 otherwise).
        :param mask2: Binary mask for the second image (1 where image2 is valid, 0 otherwise).
        :param gradient_width: Width of the gradient blend along the overlap edge (in pixels).
        :return: A blended image.
        """
        import cv2
        import numpy as np

        # cv2.imwrite("masks/mask1_f.png", mask1 * 255)

        if mask1.ndim == 3:  # If the image has multiple channels
            mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        if mask2.ndim == 3:  # If the image has multiple channels
            mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, (30,30))
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, (30,30))

        # cv2.imwrite("masks/mask1.png", mask1 * 255)
        # cv2.imwrite("masks/mask2.png", mask2 * 255)

        # Compute the overlap region
        overlap_mask = mask1 & mask2

        # cv2.imwrite("masks/overlap_mask.png", overlap_mask * 255)

        half_gradient_width = gradient_width // 2

        # Dilate the overlap edges by the specified gradient width
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gradient_width, gradient_width))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (half_gradient_width, half_gradient_width))
        # dilated_mask1 = cv2.dilate(mask1, kernel)
        # dilated_mask2 = cv2.dilate(mask2, kernel)

        # dilated_mask_crop1 = cv2.dilate(mask1, kernel2)
        # dilated_mask_crop2 = cv2.dilate(mask2, kernel2)

        erode_mask1 = cv2.erode(mask1, kernel).astype(np.uint8)
        erode_mask2 = cv2.erode(mask2, kernel).astype(np.uint8)

        erode_mask_crop1 = cv2.erode(mask1, kernel2)
        erode_mask_crop2 = cv2.erode(mask2, kernel2)

        # cv2.imwrite("masks/dilated_mask1.png", dilated_mask1 * 255)
        # cv2.imwrite("masks/dilated_mask2.png", dilated_mask2 * 255)

        # cv2.imwrite("masks/erode_mask1.png", erode_mask1 * 255)
        # cv2.imwrite("masks/erode_mask2.png", erode_mask2 * 255)


        # Restrict the gradient region to the dilated overlap area
        # gradient_region1 = (dilated_mask1 & ~mask1).astype(np.uint8)
        # gradient_region2 = (dilated_mask2 & ~mask2).astype(np.uint8)
        gradient_region1 = (~erode_mask1 & mask1).astype(np.uint8)
        gradient_region2 = (~erode_mask2 & mask2).astype(np.uint8)

        # gradient_region1 = cv2.morphologyEx(mask1, cv2.MORPH_GRADIENT, kernel)
        # gradient_region2 = cv2.morphologyEx(mask2, cv2.MORPH_GRADIENT, kernel)

        # Compute distance transforms for the overlap
        # distance_transform1 = cv2.distanceTransform(gradient_region1, cv2.DIST_L2, maskSize=5) + gradient_width / 2
        distance_transform1 = cv2.distanceTransform(gradient_region1, cv2.DIST_L2, maskSize=5)
        distance_transform2 = cv2.distanceTransform(gradient_region2, cv2.DIST_L2, maskSize=5)
        
        # shifted_distance_transform1 = distance_transform1 - (gradient_width / 2.0)
        # shifted_distance_transform2 = distance_transform2 - gradient_width / 2.0

        # shifted_distance_transform1[shifted_distance_transform1 < 0] = 0
        # shifted_distance_transform2[shifted_distance_transform2 < 0] = 0

        # cv2.imwrite("masks/gradient_region1.png", gradient_region1 * 255)
        # cv2.imwrite("masks/gradient_region2.png", gradient_region2 * 255)

        # Normalize the distance transforms to [0, 1] within the gradient width
        # weight1 = np.clip(distance_transform1 / gradient_width, 0, 1)
        weight1 = 1 - np.clip(distance_transform1 / gradient_width * 2, 0, 1)
        # weight1 = 1 - np.clip(shifted_distance_transform1 / gradient_width, 0, 1)
        # weight1 = np.clip(shifted_distance_transform1 / gradient_width, 0, 1)
        # weight2 = np.clip(distance_transform2 / gradient_width, 0, 1)
        weight2 = 1 - np.clip(distance_transform2 / gradient_width * 2, 0, 1)
        # weight2 = 1 - np.clip(shifted_distance_transform2 / gradient_width, 0, 1)
        # weight2 = np.clip(shifted_distance_transform2 / gradient_width, 0, 1)

        # weight1 = weight1 * dilated_mask_crop1
        weight1 = weight1 * erode_mask_crop1
        # weight2 = weight2 * dilated_mask_crop2
        weight2 = weight2 * erode_mask_crop2
        weight1 = np.clip((weight1 - 0.5) * 2 , 0, 1)
        weight2 = np.clip((weight2 - 0.5) * 2, 0, 1)


        # cv2.imwrite("masks/weight1.png", weight1 * 255)
        # cv2.imwrite("masks/weight2.png", weight2 * 255)

        im2mask = (mask2 - erode_mask1 - weight1) * mask2 
        im2mask[im2mask < 0] = 0
        inv_weight1 = 1 - weight1
        # cv2.imwrite("masks/inv_weight1.png", inv_weight1 * 255)
        im1mask = mask1 - inv_weight1 + (mask1 - overlap_mask)
        im1mask[im1mask > 1] = 1

        # im1mask = cv2.normalize(im1mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # im2mask = cv2.normalize(im2mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # cv2.imwrite("masks/im1mask.png", im1mask * 255)
        # cv2.imwrite("masks/im2mask.png", im2mask * 255)

        full_mask = im1mask + im2mask

        # cv2.imwrite("masks/full_mask.png", full_mask * 255)

        # blended_image = (image1 * im1mask + image2 * im2mask)
        for i in range(3):
            image1[:, :, i] = image1[:, :, i] * im1mask
            image2[:, :, i] = image2[:, :, i] * im2mask

        # cv2.imwrite("masks/image1.png", image1)
        # cv2.imwrite("masks/image2.png", image2)

        blended_image = image1 + image2

        return blended_image.astype(np.uint8)

        # Combine weights only within the overlap and gradient regions
        combined_weights = weight1 + weight2

        # cv2.imwrite("masks/combined_weights.png", combined_weights * 255)
        # combined_weights = np.clip(combined_weights, 0, 1)

        combined_weights[combined_weights == 0] = 1  # Avoid division by zero
        normalized_weight1 = weight1 / combined_weights
        normalized_weight2 = weight2 / combined_weights

        # cv2.imwrite("masks/normalized_weight1.png", normalized_weight1 * 255)
        # cv2.imwrite("masks/normalized_weight2.png", normalized_weight2 * 255)

        # Apply full intensity outside the gradient and overlap regions
        normalized_weight1[mask1 & ~overlap_mask] = 1
        normalized_weight2[mask2 & ~overlap_mask] = 1
        normalized_weight1[~mask1] = 0  # Ensure no contribution from image1 where it is invalid
        normalized_weight2[~mask2] = 0  # Ensure no contribution from image2 where it is invalid


        # cv2.imwrite("masks/normalized_weight1_full.png", normalized_weight1 * 255)
        # cv2.imwrite("masks/normalized_weight2_full.png", normalized_weight2 * 255)

        # Ensure weights match the dimensions of the images
        if len(image1.shape) == 3 and normalized_weight1.ndim == 2:
            normalized_weight1 = np.repeat(normalized_weight1[:, :, np.newaxis], image1.shape[2], axis=2)
        if len(image2.shape) == 3 and normalized_weight2.ndim == 2:
            normalized_weight2 = np.repeat(normalized_weight2[:, :, np.newaxis], image2.shape[2], axis=2)

        # Blend the images using the weights
        blended_image = (((image1.astype(np.float32) * normalized_weight1 +
                        image2.astype(np.float32) * normalized_weight2)))

        return blended_image.astype(np.uint8)


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
        increment = 0
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
                # # mask_additional = (warped_image > 0).astype(np.uint8)
                # mask_additional = self.create_gradient_mask(warped_image)
                # cv2.imwrite("mask_additional.png", mask_additional)

                # # mask_main = (final_image > 0).astype(np.uint8)
                # mask_main = self.create_gradient_mask(final_image)
                # cv2.imwrite("mask_main.png", mask_main)

                # final_image = self.gradient_blend(final_image, warped_image, mask_main, mask_additional)
                gray1 = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.medianBlur(gray1, 3)  # Replace 3 with an appropriate kernel size
                gray2 = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                # gray2 = cv2.medianBlur(gray1, 3)  # Replace 3 with an appropriate kernel size

                # cv2.imwrite("masks/gray1.png", gray1)

                mask1 = (gray1 > 0).astype(np.uint8)
                mask2 = (gray2 > 0).astype(np.uint8)
                
                # final_image = self.gradient_blend_edge_based(final_image, warped_image, (final_image > 0).astype(np.uint8), (warped_image > 0).astype(np.uint8))
                final_image = self.gradient_blend_along_edge(image1=final_image, image2=warped_image, mask1=mask1, mask2=mask2, gradient_width=400)

                # cv2.imwrite(f"out/test/final_image_{increment}.png", final_image)
            else:
                mask_additional = (warped_image > 0).astype(np.uint8)
                # mask_main = (final_image > 0).astype(np.uint8)
                mask_main = 1 - mask_additional

                final_image = (final_image * mask_main) + (warped_image * mask_additional)
            # mask = cv2.cvtColor((warped_image > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)

            # final_image = cv2.add(final_image, warped_image, mask=mask)
            increment += 1
            print(f"Image {increment} stitched.")


        return final_image

   