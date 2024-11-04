import time
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Point, Polygon


import KeypointStorage as kp

def GetImage(index, folder_path):
     # List all files in the folder, sorted by name
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg')))
    
    # Check if the index is within bounds
    if index < 0 or index >= len(files):
        raise IndexError(f"Index {index} is out of bounds. Available files: {len(files)}")

    # Get the full path of the image
    image_path = os.path.join(folder_path, files[index])

    # Open the image
    # image = Image.open(image_path)
    image = cv.imread(image_path)
    
    return image

def plot_rectangle(corners, color, label):
    """
    Plot a rectangle given its corners.

    Vstack the corners to close the rectangle and plot the lines.
    [[x1, y1]
    [x2, y2]
    [x3, y3]
    [x4, y4]
    [x1, y1]]
    """
    # Append the first point to the end to close the rectangle
    corners = np.vstack([corners, corners[0]])
    # print(corners)
    plt.plot(corners[:, 0], corners[:, 1], color=color, label=label, linewidth=2)


def GetSearchArea(Frame, ScaleFactor=2):
    """
    Get the search area for the next image based on the frame of the previous image.
    
    Parameters:
    - Frame: List of 4 2-dimenstional points ([x1,y1], [x2,y2], [x3,y3], [x4,y4]) of the bounding box.
    - ScaleFactor: Scale factor to increase the size of the search area, based on frame.
    
    Returns:
    - SearchArea: List of 4 2-dimenstional points ([x1,y1], [x2,y2], [x3,y3], [x4,y4]) of the search area.
    """
    corners = np.array(Frame).reshape(-1, 2)

    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])
    center = np.array([center_x, center_y])

    # Calculate vectors from the center to the corners
    vectors = corners - center

    scaled_vectors = vectors * ScaleFactor

    new_corners = center + scaled_vectors

    plt.figure(figsize=(10, 10))
    plt.plot(center[0], center[1], 'ro', label='Center')
    plot_rectangle(corners, 'b', 'Original Frame')
    plot_rectangle(new_corners, 'g', 'Search Area')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.title('Search Area for the Next Image')
    
    # plt.savefig('search_area.png')
    # plt.show()

    return new_corners

   

if __name__ == '__main__':

    #start timer
    start = time.time()

    if len(sys.argv) < 3:
        print("Usage: python main.py <imgs_path> <range_imgs>")
        sys.exit(1)

    imgs_path = sys.argv[1]
    range_imgs = int(sys.argv[2])

    first_img = GetImage(0, imgs_path)

    #copy first image to main canvas
    main_canvas = first_img

    first_img_gray = cv.cvtColor(first_img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(first_img_gray, None)

    ################### INITIALISE KEYPOINTS STORAGE ###################
    kp_storage = kp.KeypointStorage()
    color = random.choice(range(256)), random.choice(range(256)), random.choice(range(256))
    kp_storage.add_or_update_keypoints(keypoints1, descriptors1, color=color, iteration=0)

    previous_box = [
        [0,0], 
        [0,first_img.shape[0]], 
        [first_img.shape[1], first_img.shape[0]], 
        [first_img.shape[1], 0]
    ]

    for i in range(range_imgs):

        color = random.choice(range(256)), random.choice(range(256)), random.choice(range(256))

        additional_img = GetImage(i+1, imgs_path)

        additional_img_gray = cv.cvtColor(additional_img, cv.COLOR_BGR2GRAY)

        keypoints2, descriptors2 = sift.detectAndCompute(additional_img_gray, None)

        # rotation_matrix = np.array([
        #     [np.cos(np.pi/4), -np.sin(np.pi/4)],
        #     [np.sin(np.pi/4), np.cos(np.pi/4)]
        #     ])
        # test_box = [[0, 0], [240, 56], [320, 240], [0, 320]]
        # test_rotation = previous_box @ rotation_matrix
        # SearchArea = GetSearchArea(test_rotation, ScaleFactor=2)
        
        SearchArea = GetSearchArea(previous_box, ScaleFactor=2)
        keypoints1, descriptors1 = kp_storage.query_keypoints(SearchArea)

        #convert to numpy array
        keypoints1 = np.array(keypoints1)
        descriptors1 = np.array(descriptors1)

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        keypoints1_coords = np.array([kp["coords"] for kp in keypoints1])
        keypoints1_descriptors = np.array([kp["descriptor"] for kp in keypoints1])

        src_pts = np.float32([keypoints1_coords[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        H_inv = np.linalg.inv(H)

        second_img_h = additional_img.shape[0]
        second_img_w = additional_img.shape[1]
        corners_img2 = np.float32([[0, 0], 
                                   [0, second_img_h], 
                                   [second_img_w, second_img_h], 
                                   [second_img_w, 0]]) \
                                   .reshape(-1, 1, 2)
        
        warped_corners_img2 = cv.perspectiveTransform(corners_img2, H_inv)
        all_corners = np.concatenate((corners_img2, warped_corners_img2), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        translation_matrix = np.array([[1, 0, translation_dist[0]], 
                                       [0, 1, translation_dist[1]], 
                                       [0, 0, 1]])
        
        # Create the affine translation matrix
        translation_matrix_affine = np.array([
            [1, 0, translation_dist[0]],
            [0, 1, translation_dist[1]]
        ], dtype=np.float32)
        H_translation = translation_matrix @ H_inv
        # H_translation = H @ translation_matrix

        x_main_canvas_min = min(main_canvas.shape[1], x_min)
        y_main_canvas_min = min(main_canvas.shape[0], y_min)

        x_main_canvas_max = max(main_canvas.shape[1], x_max)
        y_main_canvas_max = max(main_canvas.shape[0], y_max)

        new_width = x_main_canvas_max - x_main_canvas_min
        new_height = y_main_canvas_max - y_main_canvas_min
        warped_additional_img = cv.warpPerspective(additional_img, H_translation, (new_width, new_height))

        mask1 = np.zeros((new_height, new_width), dtype=np.float32)
        mask2 = np.zeros((new_height, new_width), dtype=np.float32)

        x_offset = translation_dist[0]
        y_offset = translation_dist[1]

        new_main_canvas = np.zeros((new_height, new_width, 3), np.uint8)
        print("NEW MAIN CANVAS SIZE: ", new_main_canvas.shape)
        new_main_canvas[y_offset:main_canvas.shape[0]+y_offset, x_offset:main_canvas.shape[1]+x_offset] = main_canvas
        mask1[y_offset:main_canvas.shape[0]+y_offset, x_offset:main_canvas.shape[1]+x_offset] = 1

        warped_img2_gray = cv.cvtColor(warped_additional_img, cv.COLOR_BGR2GRAY)
        mask2[warped_img2_gray > 0] = 1

        alpha = mask1 / (mask1 + mask2 + 1e-10)
        beta = 1 - alpha

        for c in range(3):
            new_main_canvas[:, :, c] = (alpha * new_main_canvas[:, :, c] + beta * warped_additional_img[:, :, c]).astype(np.uint8)

        previous_box = warped_corners_img2.reshape(-1, 2)

        ################### ADD KEYPOINTS TO STORAGE ###################
        matched_keypoints = []
        matched_descriptors = []
        for match in good_matches:
            kp_data = keypoints1[match.queryIdx]
            kpoint = cv.KeyPoint(x=kp_data["coords"][0], 
                                 y=kp_data["coords"][1], 
                                 size=kp_data["scale"],
                                 angle=kp_data["angle"],
                                 response=kp_data["response"],
                                 octave=kp_data["octave"])
            
            matched_keypoints.append(kpoint)

            matched_descriptors.append(keypoints1_descriptors[match.queryIdx])

        kp_storage.add_or_update_keypoints(matched_keypoints, matched_descriptors, color=color, reliability_multiplier=1.5, iteration=i+1)
        ################### ADD KEYPOINTS TO STORAGE ###################

        ############ update realability by 0.7 to points that are not matched, but lying within the search area and new image area ########

        search_area_polygon = Polygon(SearchArea.reshape(-1, 2))
        new_image_bounding_box = cv.boundingRect(warped_corners_img2)
        new_image_polygon = Polygon(warped_corners_img2.reshape(-1, 2))

        not_matched_keypoints = []
        not_matched_descriptors = []

        new_keypoints = []
        new_descriptors = []
        for kp, ds in zip(keypoints2, descriptors2):
            if not search_area_polygon.contains(Point(kp.pt)) or not new_image_polygon.contains(Point(kp.pt)):
                not_matched_keypoints.append(kp)
                not_matched_descriptors.append(ds)
            else:
                new_keypoints.append(kp)
                new_descriptors.append(ds)

        kp_storage.add_or_update_keypoints(not_matched_keypoints, not_matched_descriptors, color=color, reliability_multiplier=0.7, iteration=i+1)

        keypoints2_coords = np.array([kp.pt for kp in new_keypoints], dtype=np.float32).reshape(-1, 1, 2)
        keypoints2_coords_transformed = cv.perspectiveTransform(keypoints2_coords, H_inv)
        keypoints2_transformed = [cv.KeyPoint(x=pt[0][0], y=pt[0][1], size=1) for pt in keypoints2_coords_transformed]
        # keypoints2_coords = np.array([kp.pt for kp in keypoints2], dtype=np.float32).reshape(-1, 1, 2)
        # keypoints2_coords_transformed = cv.perspectiveTransform(keypoints2_coords, H_inv)
        # keypoints2_transformed = [cv.KeyPoint(x=pt[0][0], y=pt[0][1], size=1) for pt in keypoints2_coords_transformed]

        kp_storage.add_or_update_keypoints(keypoints2_transformed, new_descriptors, color=color, iteration=i+1)

        visualisation = kp_storage.visualize_keypoints()
        cv.imwrite(f'keypoints_storage_{i}.png', visualisation)
        # print("PREVIOUS BOX: ", previous_box)

        main_canvas = new_main_canvas

        ################### DRAW MATCHES ###################

        # inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]

        # inlier_matches_for_draw = [[m] for m in inlier_matches]

        # matches_img = cv.drawMatchesKnn(
        #     first_img_gray, keypoints1, 
        #     additional_img_gray, keypoints2, 
        #     inlier_matches_for_draw, None, 
        #     # **draw_params
        #     flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # )

        # # save image
        # cv.imwrite(f'matches_{i}.png', matches_img)

        ################### DRAW MATCHES ###################

        # cv.imshow('Matches', matches_img)
        # cv.waitKey(0)

    cv.imwrite(f'blended_img_cnt{range_imgs+1}.png', main_canvas)

    # count runtime
    end = time.time()
    print("Runtime: ", end-start)
