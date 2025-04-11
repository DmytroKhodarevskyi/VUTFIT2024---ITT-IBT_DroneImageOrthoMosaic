import cv2
import numpy as np

def align_reference_to_stitched(stitched_path, reference_path, output_path):
    # Load images in grayscale
    stitched = cv2.imread(stitched_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    reference_original = cv2.imread(reference_path, cv2.IMREAD_COLOR)

    if reference is None or stitched is None:
        print("Error: One of the images could not be loaded.")
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(stitched, None)

    # FLANN-based matcher parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the top 2 matches for each descriptor
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe’s ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(reference, kp1, stitched, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the image with matches
    # cv2.imshow("Feature Matches", img_matches)
    # cv2.waitKey(0)  # Wait for key press to close
    # cv2.destroyAllWindows()
    cv2.imwrite("matches.png", img_matches)

    # Ensure enough matches are found
    if len(good_matches) > 10:
        # Extract point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute Homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        H_inv = np.linalg.inv(H)

        # Warp the reference image to match the stitched image perspective
        height, width = stitched.shape
        # aligned_reference = cv2.warpPerspective(reference_original, H_inv, (width, height))
        aligned_reference = cv2.warpPerspective(reference_original, H, (width, height))

        # Save the aligned reference image
        cv2.imwrite(output_path, aligned_reference)

        print(f"Aligned reference image saved as {output_path}")

    else:
        print("Not enough good matches to align images.")

if __name__ == "__main__":
    # Example paths (update these paths as needed)
    # stitched_image_path = "blended_img_cnt31.png"
    # stitched_image_path = "./out/blended/golf/blended_img_cnt31.png"
    stitched_image_path = "./src/highway_source_56.png"
    # reference_image_path = "golf_reference_30.png"
    reference_image_path = "./src/highway_reference2_56.png"
    output_image_path = "aligned_reference_2.png"

    # Call the function to align the reference image to the stitched image
    align_reference_to_stitched(stitched_image_path, reference_image_path, output_image_path)

# Example usage
# align_reference_to_stitched("stitched_image.jpg", "cropped_reference.jpg", "aligned_reference.jpg")
