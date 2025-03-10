import numpy as np
import cv2 as cv2
import os
import json
# import ImageData

class ImageData:
    # def __init__(self, image, homography_matrix=None, overall_transform_matrix=None):
    def __init__(self, image, image_name, homography_matrix=None):
        """
        Initializes an ImageData object.

        :param image: The image (numpy array or other representation).
        :param homography_matrix: Homogeneous transformation matrix for this image (default is identity matrix).
        :param overall_transform_matrix: Overall transformation matrix (default is identity matrix).
        """
        self.image = image
        self.image_name = image_name
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.homography_matrix = homography_matrix if homography_matrix is not None else np.eye(3)
        # self.overall_transform_matrix = overall_transform_matrix if overall_transform_matrix is not None else np.eye(3)

    def set_homography_matrix(self, matrix):
        """
        Sets the homography matrix.

        :param matrix: A 3x3 numpy array representing the new homography matrix.
        """
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 matrix.")
        self.homography_matrix = matrix

    # def set_overall_transform_matrix(self, matrix):
    #     """
    #     Sets the overall transformation matrix.

    #     :param matrix: A 3x3 numpy array representing the new overall transformation matrix.
    #     """
    #     if matrix.shape != (3, 3):
    #         raise ValueError("Overall transform matrix must be a 3x3 matrix.")
    #     self.overall_transform_matrix = matrix

    def get_homography_matrix(self):
        """Returns the homography matrix."""
        return self.homography_matrix

    # def get_overall_transform_matrix(self):
    #     """Returns the overall transformation matrix."""
    #     return self.overall_transform_matrix

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

    # def add_image(self, image, homography_matrix=None, overall_transform_matrix=None):
    def add_image(self, image, image_name, homography_matrix=None):
        """
        Adds a new image and its associated matrices to the collection.

        :param image: The image (numpy array or other representation).
        :param homography_matrix: Homogeneous transformation matrix for this image (default is identity matrix).
        :param overall_transform_matrix: Overall transformation matrix (default is identity matrix).
        """
        # image_data = ImageData(image, homography_matrix, overall_transform_matrix)
        image_data = ImageData(image, image_name, homography_matrix)
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

    # def get_all_overall_transform_matrices(self):
    #     """
    #     Retrieves all overall transformation matrices in the collection.

    #     :return: A list of all overall transformation matrices.
    #     """
    #     return [image_data.get_overall_transform_matrix() for image_data in self.images_data]
    def save_homography_data(self, path):
        """
        Save each photo homography matrix to a separate file with its photo filename.
        Then could be used to check if the homography for this photo already exists.
        """
        for image_data in self.images_data:
            np.savetxt(path + image_data.image_name + '.txt', image_data.homography_matrix)

    def load_homography_data(self, path, name):
        """
        Load homography matrix from file.
        """
        filepath = path + name + '.txt'
        try:
            data = np.loadtxt(filepath)
        except:
            return None

        return data

    def save_image_data(self, image_name, H, good_matches, keypoints2, descriptors2, save_path):
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

        # Save homography as plain text
        np.savetxt(os.path.join(save_path, f"{image_name}_homography.txt"), H)

        # Convert good_matches (cv2.DMatch objects) into dictionaries
        good_matches_data = [{"queryIdx": m.queryIdx, "trainIdx": m.trainIdx, "imgIdx": m.imgIdx, "distance": m.distance} for m in good_matches]
        
        # Convert keypoints2 (cv2.KeyPoint objects) into dictionaries
        keypoints2_data = [{"x": kp.pt[0], "y": kp.pt[1], "size": kp.size, "angle": kp.angle, 
                            "response": kp.response, "octave": kp.octave, "class_id": kp.class_id} for kp in keypoints2]

        # Save matches and keypoints as JSON
        with open(os.path.join(save_path, f"{image_name}_good_matches.json"), "w") as f:
            json.dump(good_matches_data, f)

        with open(os.path.join(save_path, f"{image_name}_keypoints2.json"), "w") as f:
            json.dump(keypoints2_data, f)

        # Save descriptors as binary
        np.save(os.path.join(save_path, f"{image_name}_descriptors2.npy"), descriptors2)

        print(f"✅ Saved image data for {image_name}")

    def load_image_data(self, image_name, load_path):
        try:
            # Load homography from text file
            H = np.loadtxt(os.path.join(load_path, f"{image_name}_homography.txt"))

            # Load matches and keypoints from JSON
            with open(os.path.join(load_path, f"{image_name}_good_matches.json"), "r") as f:
                good_matches_data = json.load(f)

            with open(os.path.join(load_path, f"{image_name}_keypoints2.json"), "r") as f:
                keypoints2_data = json.load(f)

            # Convert dictionaries back to cv2.DMatch objects
            good_matches = [cv2.DMatch(m["queryIdx"], m["trainIdx"], m["imgIdx"], m["distance"]) for m in good_matches_data]

            # Convert dictionaries back to cv2.KeyPoint objects
            keypoints2 = [cv2.KeyPoint(kp["x"], kp["y"], kp["size"], kp["angle"], kp["response"], kp["octave"], kp["class_id"]) for kp in keypoints2_data]

            # Load descriptors from binary file
            descriptors2 = np.load(os.path.join(load_path, f"{image_name}_descriptors2.npy"))

            return H, good_matches, keypoints2, descriptors2

        except FileNotFoundError:
            print(f"⚠️  Missing saved data for {image_name}. Recomputing...")
            return None
        except json.JSONDecodeError:
            print(f"⚠️  Corrupt JSON file detected for {image_name}. Recomputing...")
            return None



