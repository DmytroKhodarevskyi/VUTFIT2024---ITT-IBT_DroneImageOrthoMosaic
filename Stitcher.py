import ImageStitchingData
import cv2 
import numpy as np

class Stitcher:
    # def __init__(self, data=None):
    #     """
    #     Initializes a Stitcher object.
        
    #     :param data: Optional. An existing ImageStitchingData object to set. If None, a new instance is created.
    #     """
    #     if data is not None:
    #         self.data = data
    #     else:
    #         self.data = ImageStitchingData()

    # def set_data(self, data):
    #     """
    #     Sets the ImageStitchingData object.

    #     :param data: The ImageStitchingData object to set.
    #     """
    #     self.data = data

    # def stitch_images(self):
    #     final = data.images_data[0].get_image()

    #     for i in range(1, len(data.images_data)):
    #         additional = data.images_data[i].get_image()

    #         h_main, w_main = final.shape[:2]
    #         h_add, w_add = additional_img.shape[:2]
        
    #         corners_main = np.array([[0, 0], [w_main, 0], [w_main, h_main], [0, h_main]], dtype=np.float32).reshape(-1, 1, 2)

    #         corners_add = np.array([[0, 0], [w_add, 0], [w_add, h_add], [0, h_add]], dtype=np.float32).reshape(-1, 1, 2)

    #         transformed_corners = cv2.perspectiveTransform(corners_add, data.images_data[i].get_homography_matrix())

    #         warped_additional = cv2.warpPerspective(additional, data.images_data[i].get_homography_matrix(), (first.shape[1] + additional.shape[1], first.shape[0]))

    def __init__(self, stitching_data):
        """
        Initializes a Stitcher object.

        :param stitching_data: An ImageStitchingData object containing images and their transformation matrices.
        """
        self.stitching_data = stitching_data

    def stitch_images(self):
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

            mask_additional = (warped_image > 0).astype(np.uint8)
            # mask_main = (final_image > 0).astype(np.uint8)
            mask_main = 1 - mask_additional

            final_image = (final_image * mask_main) + (warped_image * mask_additional)
            # mask = cv2.cvtColor((warped_image > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)

            # final_image = cv2.add(final_image, warped_image, mask=mask)

        return final_image