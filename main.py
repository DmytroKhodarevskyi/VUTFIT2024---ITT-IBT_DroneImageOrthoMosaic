import time
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Point, Polygon


import KeypointStorage as kp
import ImageData as img_data
import ImageStitchingData as imgs_data
import Stitcher as stitcher

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

    # cumulative_H = np.eye(3)
    # overall_translation = np.eye(3)

    cumulative_transform = np.eye(3)


    #start timer
    start = time.time()

    if len(sys.argv) < 3:
        print("Usage: python main.py <imgs_path> <range_imgs>")
        sys.exit(1)

    imgs_path = sys.argv[1]
    range_imgs = int(sys.argv[2])

    image_storage = imgs_data.ImageStitchingData()
    first_img = GetImage(0, imgs_path)
    image_storage.add_image(first_img, overall_transform_matrix=np.eye(3), homography_matrix=np.eye(3))

    #copy first image to main canvas
    main_canvas = first_img

    # first_img_gray = cv.cvtColor(first_img, cv.COLOR_BGR2GRAY)
    first_img_gray = image_storage.images_data[0].gray_image

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

        if np.linalg.det(H) == 0:
            print("Homography is singular")
            print("Iteration: ", i+1)
            continue

        H_inv = np.linalg.inv(H)

        # image_storage.add_image(additional_img, homography_matrix=H_inv, overall_transform_matrix=cumulative_transform)

        h_add, w_add = additional_img.shape[:2]
        h_main, w_main = main_canvas.shape[:2]
        
        corners_add = np.array([[0, 0], [w_add, 0], [w_add, h_add], [0, h_add]], dtype=np.float32).reshape(-1, 1, 2)
        corners_main = np.array([[0, 0], [w_main, 0], [w_main, h_main], [0, h_main]], dtype=np.float32).reshape(-1, 1, 2)
        
        # transformed_corners_add = cv.perspectiveTransform(corners_add, H_inv)
        transformed_corners_add = cv.perspectiveTransform(corners_add, cumulative_transform @ H_inv)
        all_corners = np.concatenate((corners_main, transformed_corners_add), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation_matrix = np.array([[1, 0, -x_min],
                                    [0, 1, -y_min],
                                    [0, 0, 1]], dtype=np.float32)

        cumulative_transform = translation_matrix @ cumulative_transform

        out_width = x_max - x_min
        out_height = y_max - y_min

        image_storage.add_image(additional_img, homography_matrix=H_inv, overall_transform_matrix=cumulative_transform)

        # # warped_additional = cv.warpPerspective(additional_img, translation_matrix @ H_inv, (out_width, out_height))
        # warped_additional = cv.warpPerspective(additional_img, cumulative_transform @ H_inv, (out_width, out_height))
        # translated_main_canvas = cv.warpPerspective(main_canvas, translation_matrix, (out_width, out_height))

        # mask_additional = (warped_additional > 0).astype(np.uint8)
        # # mask_main_canvas = (translated_main_canvas > 0).astype(np.uint8)
        # mask_main_canvas = 1 - mask_additional


        # combined_mask = mask_additional + mask_main_canvas
        # combined_mask[combined_mask == 0] = 1

        # # blended_img = (warped_additional + translated_main_canvas) / combined_mask
        # # blended = ((translated_main_canvas.astype(np.float32) + warped_additional.astype(np.float32)) / combined_mask).astype(np.uint8)
        # # overlapped = (translated_main_canvas * mask_main_canvas) + (warped_additional * mask_additional)
        # blended = (translated_main_canvas * mask_main_canvas) + (warped_additional * mask_additional)


        # overlapped = (translated_main_canvas * mask_main_canvas) + (warped_additional * mask_additional)


        # cv.imwrite(f'out/blended/golf/blended_img_test_{i}.png', blended)

        # previous_box = warped_corners_img2.reshape(-1, 2)
        previous_box = transformed_corners_add.reshape(-1, 2)

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
        # new_image_bounding_box = cv.boundingRect(warped_corners_img2)
        # new_image_polygon = Polygon(warped_corners_img2.reshape(-1, 2))
        new_image_polygon = Polygon(transformed_corners_add.reshape(-1, 2))

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
        # cv.imwrite(f'out/keypoints/golf/keypoints_storage_{i}.png', visualisation)
        cv.imwrite(f'out/keypoints/highway/keypoints_storage_{i}.png', visualisation)
        # print("PREVIOUS BOX: ", previous_box)

        # main_canvas = new_main_canvas
        print("Iteration: ", i+1)
        # main_canvas = blended


    Sticher = stitcher.Stitcher(image_storage)
    final_image = Sticher.stitch_images()


    # cv.imwrite(f'out/blended/golf/blended_img_cnt{range_imgs+1}.png', main_canvas)

    cv.imwrite(f'out/blended/highway/blended_img_cnt_new_{range_imgs+1}.png', final_image)
    # cv.imwrite(f'out/blended/highway/blended_img_cnt{range_imgs+1}.png', main_canvas)

    # count runtime
    end = time.time()
    print("Runtime: ", end-start)
