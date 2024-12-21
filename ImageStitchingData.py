import numpy as np
import cv2 as cv2
# import ImageData

class ImageData:
    def __init__(self, image, homography_matrix=None, overall_transform_matrix=None):
        """
        Initializes an ImageData object.

        :param image: The image (numpy array or other representation).
        :param homography_matrix: Homogeneous transformation matrix for this image (default is identity matrix).
        :param overall_transform_matrix: Overall transformation matrix (default is identity matrix).
        """
        self.image = image
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.homography_matrix = homography_matrix if homography_matrix is not None else np.eye(3)
        self.overall_transform_matrix = overall_transform_matrix if overall_transform_matrix is not None else np.eye(3)

    def set_homography_matrix(self, matrix):
        """
        Sets the homography matrix.

        :param matrix: A 3x3 numpy array representing the new homography matrix.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 matrix.")
        self.homography_matrix = matrix

    def set_overall_transform_matrix(self, matrix):
        """
        Sets the overall transformation matrix.

        :param matrix: A 3x3 numpy array representing the new overall transformation matrix.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Overall transform matrix must be a 3x3 matrix.")
        self.overall_transform_matrix = matrix

    def get_homography_matrix(self):
        """Returns the homography matrix."""
        return self.homography_matrix

    def get_overall_transform_matrix(self):
        """Returns the overall transformation matrix."""
        return self.overall_transform_matrix

    def get_image(self):
        """Returns the image."""
        return self.image

class ImageStitchingData:
    def __init__(self):
        """
        Initializes an ImageStitchingData object to store multiple images and their associated matrices.
        """
        self.images_data = []
        self

    def add_image(self, image, homography_matrix=None, overall_transform_matrix=None):
        """
        Adds a new image and its associated matrices to the collection.

        :param image: The image (numpy array or other representation).
        :param homography_matrix: Homogeneous transformation matrix for this image (default is identity matrix).
        :param overall_transform_matrix: Overall transformation matrix (default is identity matrix).
        """
        image_data = ImageData(image, homography_matrix, overall_transform_matrix)
        self.images_data.append(image_data)

    def get_image_data(self, index):
        """
        Retrieves the ImageData object at the specified index.

        :param index: The index of the image data to retrieve.
        :return: An ImageData object.
        """
        if index < 0 or index >= len(self.images_data):
            raise IndexError("Index out of range.")
        return self.images_data[index]

    def get_all_images(self):
        """
        Retrieves all images in the collection.

        :return: A list of all images.
        """
        return [image_data.get_image() for image_data in self.images_data]

    def get_all_homography_matrices(self):
        """
        Retrieves all homography matrices in the collection.

        :return: A list of all homography matrices.
        """
        return [image_data.get_homography_matrix() for image_data in self.images_data]

    def get_all_overall_transform_matrices(self):
        """
        Retrieves all overall transformation matrices in the collection.

        :return: A list of all overall transformation matrices.
        """
        return [image_data.get_overall_transform_matrix() for image_data in self.images_data]
