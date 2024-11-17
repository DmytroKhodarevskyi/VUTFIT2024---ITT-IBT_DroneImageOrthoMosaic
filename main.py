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

    cumulative_H = np.eye(3)
    overall_translation = np.eye(3)

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

        H, status = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if np.sum(status) == 0:
            print("No inliers found in RANSAC")
            print("Iteration: ", i+1)
            continue

        if np.all(H == 0):
            print("No homography found")
            print("Iteration: ", i+1)
            continue
        # H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3.0)

        if np.linalg.det(H) == 0:
            print("Homography is singular")
            print("Iteration: ", i+1)
            continue

        H_inv = np.linalg.inv(H)

        second_img_h = additional_img.shape[0]
        second_img_w = additional_img.shape[1]

        corners_img2 = np.float32([[0, 0], 
                                   [0, second_img_h], 
                                   [second_img_w, second_img_h], 
                                   [second_img_w, 0]]) \
                                   .reshape(-1, 1, 2)

        warped_corners_img2 = cv.perspectiveTransform(corners_img2, cumulative_H @ H_inv)
        main_canvas_corners = np.float32([[0, 0], 
                                          [0, main_canvas.shape[0]], 
                                          [main_canvas.shape[1], main_canvas.shape[0]], 
                                          [main_canvas.shape[1], 0]]) \
                                          .reshape(-1, 1, 2)
        
        all_corners = np.concatenate((corners_img2, warped_corners_img2), axis=0)
        # all_corners = np.concatenate((main_canvas_corners, warped_corners_img2), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]

        # x_min += overall_translation[0, 2]
        # y_min += overall_translation[1, 2]

        # x_max += overall_translation[0, 2]
        # y_max += overall_translation[1, 2]

        translation_matrix = np.array([[1, 0, translation_dist[0]], 
                                       [0, 1, translation_dist[1]], 
                                       [0, 0, 1]])
        
        overall_translation = overall_translation @ translation_matrix
        
        # H_translation = cumulative_H @ translation_matrix @ H_inv
        # H_translation = translation_matrix @ cumulative_H @ H_inv
        # H_translation = cumulative_H @ H_inv @ translation_matrix 

        H_translation = overall_translation @ H_inv

        # warped_corners_with_translation = cv.perspectiveTransform(corners_img2, H_translation)

        # all_corners_with_translation = np.concatenate((main_canvas_corners, warped_corners_with_translation), axis=0)

        # [new_x_min, new_y_min] = np.int32(all_corners_with_translation.min(axis=0).ravel() - 0.5)
        # [new_x_max, new_y_max] = np.int32(all_corners_with_translation.max(axis=0).ravel() + 0.5)

        cumulative_H = H_translation
        # Normalize the cumulative homography
        if not np.isclose(cumulative_H[2, 2], 1.0, atol=1e-6):
            cumulative_H /= cumulative_H[2, 2]
            # print("Normalized Cumulative H:", cumulative_H)

        # Extract rotation and scaling (upper-left 2x2 matrix)
        # rotation_scaling = cumulative_H[:2, :2]

        # Extract translation (last column)
        # translation = cumulative_H[:2, 2]

        # print("STITCHING IMAGE: ", i+1)

        # print("CUMULATIVE H:")
        # print(cumulative_H)
        # print("Rotation & Scaling:")
        # print(rotation_scaling)
        # print("Translation:")
        # print(translation)
        # theta = np.arctan2(rotation_scaling[1, 0], rotation_scaling[0, 0]) * 180 / np.pi
        # print("Rotation Angle (degrees):", theta)


        # print("TRANSLATION MATRIX:") 
        # print(translation_matrix)
        # print("H INV: ") 
        # print(H_inv)

        # H_translation =  H_inv
        # H_translation = H @ translation_matrix
        # new_warped_corners_img2 = cv.perspectiveTransform(corners_img2, cumulative_H)
        # all_corners = np.concatenate((corners_img2, new_warped_corners_img2), axis=0)

        # [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        # [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        x_main_canvas_min = min(main_canvas.shape[1], x_min)
        y_main_canvas_min = min(main_canvas.shape[0], y_min)

        x_main_canvas_max = max(main_canvas.shape[1], x_max)
        y_main_canvas_max = max(main_canvas.shape[0], y_max)

        # x_main_canvas_min = min(main_canvas.shape[1], new_x_min)
        # y_main_canvas_min = min(main_canvas.shape[0], new_y_min)

        # x_main_canvas_max = max(main_canvas.shape[1], new_x_max)
        # y_main_canvas_max = max(main_canvas.shape[0], new_y_max)

        # print("x_main_canvas_min: ", x_main_canvas_min)
        # print("y_main_canvas_min: ", y_main_canvas_min)
        # print("x_main_canvas_max: ", x_main_canvas_max)
        # print("y_main_canvas_max: ", y_main_canvas_max)

        new_width = x_main_canvas_max - x_main_canvas_min
        new_height = y_main_canvas_max - y_main_canvas_min
        # warped_additional_img = cv.warpPerspective(additional_img, H_translation, (new_width, new_height))
        warped_additional_img = cv.warpPerspective(additional_img, cumulative_H, (new_width, new_height))

        mask1 = np.zeros((new_height, new_width), dtype=np.float32)
        mask2 = np.zeros((new_height, new_width), dtype=np.float32)

        x_offset = translation_dist[0]
        y_offset = translation_dist[1]

        new_main_canvas = np.zeros((new_height, new_width, 3), np.uint8)
        print("NEW MAIN CANVAS SIZE: ", new_main_canvas.shape)
        print("ITERATION: ", i+1)
        print(" ")
        new_main_canvas[y_offset:main_canvas.shape[0]+y_offset, x_offset:main_canvas.shape[1]+x_offset] = main_canvas
        # mask1[y_offset:main_canvas.shape[0]+y_offset, x_offset:main_canvas.shape[1]+x_offset] = 1
        
        #set mask only where main_canvas is not black
        mask1[y_offset:main_canvas.shape[0]+y_offset, x_offset:main_canvas.shape[1]+x_offset] = cv.cvtColor(main_canvas, cv.COLOR_BGR2GRAY) > 0

        warped_img2_gray = cv.cvtColor(warped_additional_img, cv.COLOR_BGR2GRAY)
        mask2[warped_img2_gray > 0] = 1

        # if photos overlay result will be 0.5 for both masks, blending will be 50/50
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

        if keypoints2_coords_transformed is None:
            print("ERROR: keypoints2_coords_transformed is None, cannot add to storage")
        else:
            keypoints2_transformed = [cv.KeyPoint(x=pt[0][0], y=pt[0][1], size=1) for pt in keypoints2_coords_transformed]
            kp_storage.add_or_update_keypoints(keypoints2_transformed, new_descriptors, color=color, iteration=i+1)
        
        # keypoints2_transformed = [cv.KeyPoint(x=pt[0][0], y=pt[0][1], size=1) for pt in keypoints2_coords_transformed]
        # if keypoints2_transformed is None:
        #     print("ERROR: keypoints2_transformed is None, cannot add to storage")
        # else:
        #     kp_storage.add_or_update_keypoints(keypoints2_transformed, new_descriptors, color=color, iteration=i+1)
        # keypoints2_coords = np.array([kp.pt for kp in keypoints2], dtype=np.float32).reshape(-1, 1, 2)
        # keypoints2_coords_transformed = cv.perspectiveTransform(keypoints2_coords, H_inv)
        # keypoints2_transformed = [cv.KeyPoint(x=pt[0][0], y=pt[0][1], size=1) for pt in keypoints2_coords_transformed]


        visualisation = kp_storage.visualize_keypoints()
        cv.imwrite(f'out/keypoints/golf/keypoints_storage_{i}.png', visualisation)
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

    cv.imwrite(f'out/blended/golf/blended_img_cnt{range_imgs+1}.png', main_canvas)

    # count runtime
    end = time.time()
    print("Runtime: ", end-start)
