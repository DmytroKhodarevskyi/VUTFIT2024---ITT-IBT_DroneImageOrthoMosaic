import time
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Point, Polygon
from sklearn.metrics import roc_auc_score

import KeypointStorage as kp
import ImageStitchingData as imgs_data
import Stitcher as stitcher

positive_multiplier = 1.7
negative_multiplier = 0.6

HOMOGRAPHIES_PATH = "./homographies"

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
    image_name = files[index]
    
    return image, image_name

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

def GetSearchArea(Frame, ScaleFactor=1.5):
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

    return new_corners


def print_usage():
    print("Usage: python main.py <imgs_path> <range_imgs> <result_path> [-s <image_data_path>] [-h <image_data_path>] [-m <kp_map_path>] [-b <1-3>]")
    print("")
    print("Arguments:")
    print("  imgs_path        Folder where images are stored.")
    print("  range_imgs       Integer number of images to process.")
    print("  result_path      Folder to save the resulting stitched image.")
    print("Optional:")
    print("  -s <path>        Save metadata for images to <path> (cannot combine with -h).")
    print("  -h <path>        Load metadata for images from <path> (cannot combine with -s).")
    print("  -m <path>        Save keypoint maps to <path>.")
    print("  -b <1|2|3>       Blending mode (default is 1 if not specified).")
    print("")
    print("Notes:")
    print("  - Either -s or -h must be used, but not both.")
    print("  - -m and -b are optional.")
    print("  - In -b, 1 is for blending with overlap, 2 is for 50/50 mixing, and 3 is for gradient blending.")

def parse_arguments():
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)

    imgs_path = sys.argv[1]
    try:
        range_imgs = int(sys.argv[2])
    except ValueError:
        print("[ERROR] range_imgs must be an integer.")
        print_usage()
        sys.exit(1)
    result_path = sys.argv[3]

    # Default options
    save_metadata = False
    use_metadata = False
    image_data_path = None
    save_kp_map = False
    kp_map_path = None

    blending_mode = 1  # Default blending mode

    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "-s":
            if i + 1 >= len(sys.argv):
                print("[ERROR] -s requires a <image_data_path> argument.")
                print_usage()
                sys.exit(1)
            save_metadata = True
            image_data_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "-h":
            if i + 1 >= len(sys.argv):
                print("[ERROR] -h requires a <image_data_path> argument.")
                print_usage()
                sys.exit(1)
            use_metadata = True
            image_data_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "-m":
            if i + 1 >= len(sys.argv):
                print("[ERROR] -m requires a <kp_map_path> argument.")
                print_usage()
                sys.exit(1)
            save_kp_map = True
            kp_map_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "-b":
            if i + 1 >= len(sys.argv):
                print("Error: -b requires a number 1-3.")
                sys.exit(1)
            try:
                blending_mode = int(sys.argv[i + 1])
                if blending_mode not in (1, 2, 3):
                    print("Error: blending mode must be 1, 2, or 3.")
                    sys.exit(1)
            except ValueError:
                print("Error: blending mode must be an integer 1-3.")
                sys.exit(1)
            i += 2
        else:
            print(f"[ERROR] Unknown argument: {sys.argv[i]}")
            print_usage()
            sys.exit(1)

    # Check for conflict: cannot save and load at the same time
    if save_metadata and use_metadata:
        print("[ERROR] Cannot use both -s and -h at the same time.")
        sys.exit(1)

    return imgs_path, range_imgs, result_path, save_metadata, use_metadata, image_data_path, save_kp_map, kp_map_path, blending_mode

if __name__ == '__main__':

    #start timer
    start = time.time()

    roc_auc_scores_all = []
    average_scores = []

    (imgs_path, range_imgs, result_path,
     save_metadata, use_homographies, HOMOGRAPHIES_PATH,
      save_kp_map, kp_map_path, blending_mode) = parse_arguments()

    image_storage = imgs_data.ImageStitchingData()
    first_img, image_name = GetImage(0, imgs_path)
    image_storage.add_image(first_img, image_name, homography_matrix=np.eye(3))

    #copy first image to main canvas
    main_canvas = first_img

    first_img_gray = image_storage.images_data[0].gray_image

    sift = cv.SIFT_create(nOctaveLayers=5)

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

    try:
        for i in range(range_imgs):

            color = random.choice(range(256)), random.choice(range(256)), random.choice(range(256))

            additional_img, image_name = GetImage(i+1, imgs_path)

            H = np.eye(3)
            mask = np.zeros(first_img_gray.shape, dtype=np.uint8)
            good_matches = []
            keypoints2 = []
            descriptors2 = []

            image_data = None

            if use_homographies:
                image_data = image_storage.load_image_data(image_name, HOMOGRAPHIES_PATH)

            if image_data is not None:
                H, mask, good_matches, keypoints2, descriptors2 = image_data

            kp_storage.add_previous_box(previous_box)
            SearchArea = GetSearchArea(previous_box, ScaleFactor=1.5)
            keypoints1, descriptors1 = kp_storage.query_keypoints(SearchArea)

            #convert to numpy array
            keypoints1 = np.array(keypoints1)
            descriptors1 = np.array(descriptors1)

            keypoints1_coords = np.array([kp["coords"] for kp in keypoints1])
            keypoints1_descriptors = np.array([kp["descriptor"] for kp in keypoints1])

            if use_homographies and image_data is not None:
                image_storage.add_image(additional_img, image_name, homography_matrix=H)


            else:
                additional_img_gray = cv.cvtColor(additional_img, cv.COLOR_BGR2GRAY)

                keypoints2, descriptors2 = sift.detectAndCompute(additional_img_gray, None)

                matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
                matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
                

                src_pts = np.float32([keypoints1_coords[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H_current, status = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                if np.sum(status) == 0:
                    print("No inliers found in RANSAC")
                    print("Iteration: ", i+1)
                    continue

                if np.all(H_current == 0):
                    print("No homography found")
                    print("Iteration: ", i+1)
                    continue

                if np.linalg.det(H_current) == 0:
                    print("Homography is singular")
                    print("Iteration: ", i+1)
                    continue

                H = np.linalg.inv(H_current)
                mask = status.ravel()

                image_storage.add_image(additional_img, image_name, homography_matrix=H)
                if save_metadata:
                    # Save the image data to the specified path
                    image_storage.save_image_data(image_name, H, status, good_matches, keypoints2, descriptors2, HOMOGRAPHIES_PATH)


            new_image_corners = np.array([
                [0, 0],
                [additional_img.shape[1], 0],
                [additional_img.shape[1], additional_img.shape[0]],
                [0, additional_img.shape[0]]
            ], dtype=np.float32).reshape(-1, 1, 2)

            transformed_corners_add = cv.perspectiveTransform(new_image_corners, H)

            previous_box = transformed_corners_add.reshape(-1, 2)

            # Generate pseudo "probabilities" based on match distances
            match_distances = [m.distance for m in good_matches]
            normalized_distances = (np.array(match_distances) - min(match_distances)) / (max(match_distances) - min(match_distances))

            # Compute AUC
            auc_score = roc_auc_score(mask, 1 - normalized_distances)  # Higher score = better match quality
            roc_auc_scores_all.append(auc_score)

            if len(roc_auc_scores_all) % 5 == 0:
                average_score = np.mean(roc_auc_scores_all)
                print(f"Average AUC Score: {average_score:.3f}")
                average_scores.append(round(average_score, 3))


            ################### ADD KEYPOINTS TO STORAGE ###################

            ############ update realability by positive to points that are matched ########

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


            kp_storage.add_or_update_keypoints(matched_keypoints, matched_descriptors, color=color, reliability_multiplier=positive_multiplier, iteration=i+1)


            ############ update realability by 0.7 to points that are not matched, but lying within the search area and new image area ########

            search_area_polygon = Polygon(SearchArea.reshape(-1, 2))

            kp_storage.add_search_area(SearchArea)
            new_image_polygon = Polygon(transformed_corners_add.reshape(-1, 2))
            kp_storage.add_new_image_polygon(transformed_corners_add.reshape(-1, 2))

            not_matched_keypoints = []
            not_matched_descriptors = []

            new_keypoints = []
            new_descriptors = []
            for kp, ds in zip(keypoints2, descriptors2):

                kp_coords = np.array([kp.pt], dtype=np.float32).reshape(-1, 1, 2)
                transformed_coords = cv.perspectiveTransform(kp_coords, H)
                transformed_point = Point(transformed_coords[0][0][0], transformed_coords[0][0][1])

                new_kp = cv.KeyPoint(x=transformed_coords[0][0][0], 
                                    y=transformed_coords[0][0][1], 
                                    size=kp.size,
                                    angle=kp.angle,
                                    response=kp.response,
                                    octave=kp.octave)

                if not search_area_polygon.contains(transformed_point) or not new_image_polygon.contains(transformed_point):
                    not_matched_keypoints.append(new_kp)
                    not_matched_descriptors.append(ds)
                else:
                    new_keypoints.append(new_kp)
                    new_descriptors.append(ds)

            kp_storage.add_or_update_keypoints(not_matched_keypoints, not_matched_descriptors, color=color, reliability_multiplier=negative_multiplier, iteration=i+1)

            kp_storage.add_or_update_keypoints(new_keypoints, new_descriptors, color=color, iteration=i+1)
            
            ################### ADD KEYPOINTS TO STORAGE ###################

            visualisation = kp_storage.visualize_keypoints()

            if save_kp_map:
                print("Saving keypoint map to: ", kp_map_path)
                if not os.path.exists(kp_map_path):
                    os.makedirs(kp_map_path)
                cv.imwrite(os.path.join(kp_map_path, f'keypoints_storage_{i}.png'), visualisation)

            print("Iteration:", i+1)
            print("")

        

    except KeyboardInterrupt:
        if save_metadata:
            image_storage.save_image_data(image_name, H, mask, good_matches, keypoints2, descriptors2, HOMOGRAPHIES_PATH)
            print()
            print("KeyboardInterrupt: Homographies saved to ", HOMOGRAPHIES_PATH)
            exit(1)

    Sticher = stitcher.Stitcher(image_storage)
    final_image = None
    if blending_mode == 1:
        print("Blending with overlap")
        final_image = Sticher.stitch_images()
    elif blending_mode == 2:
        print("Blending with 50/50 mixing")
        final_image = Sticher.stitch_images(blending=True)
    elif blending_mode == 3:
        print("Blending with gradient")
        final_image = Sticher.stitch_images(gradient=True)
    else:
        print("Error: Invalid blending mode. Must be 1, 2, or 3.")
        sys.exit(1)

    cv.imwrite(os.path.join(result_path, f'blended_img_cnt{range_imgs+1}.png'), final_image)
    print("Final image saved to: ", os.path.join(result_path, f'blended_img_cnt{range_imgs+1}.png'))

    # count runtime
    end = time.time()
    result_time = end-start

    if result_time > 60:
        minutes = result_time // 60
        seconds = result_time % 60

        print()
        print(f"Runtime: {int(minutes)} minutes and {int(seconds)} seconds")
    else:
        print()
        print("Runtime: ", round(result_time, 3) , "seconds")

    print("Average AUC Scores: ", average_scores)
