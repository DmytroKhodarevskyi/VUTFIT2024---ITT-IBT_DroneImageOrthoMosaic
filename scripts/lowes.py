import numpy as np
import matplotlib.pyplot as plt

# Generate uniform random 2D query descriptors (query vectors)
np.random.seed(np.random.randint(0, 100))
query_descriptors = np.random.uniform(0, 1, (5, 2))  # 5 query descriptors

# Generate uniform random 2D keypoint descriptors (keypoints to match against)
keypoint_descriptors = np.random.uniform(0, 1, (10, 2))  # 10 keypoint descriptors

# # Generate random 2D query descriptors (query vectors)
# np.random.seed(np.random.randint(0, 100))
# query_descriptors = np.random.rand(5, 2)  # 5 query descriptors

# # Generate random 2D keypoint descriptors (keypoints to match against)
# keypoint_descriptors = np.random.rand(10, 2)  # 10 keypoint descriptors

# Function to compute Euclidean distance
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Find the two nearest neighbors for each query descriptor
all_matches = []
nearest_neighbors = []
for query in query_descriptors:
    distances = [euclidean_distance(query, kp) for kp in keypoint_descriptors]
    sorted_indices = np.argsort(distances)
    best_match = sorted_indices[0]
    second_best_match = sorted_indices[1]
    all_matches.append((
        distances[best_match],  # Distance to the best match
        distances[second_best_match],  # Distance to the second-best match
    ))
    nearest_neighbors.append((query, keypoint_descriptors[best_match], keypoint_descriptors[second_best_match]))

# Apply Lowe's ratio test
ratio_threshold = 0.75
good_matches_coords = []
failed_matches_coords = []

for i, match in enumerate(all_matches):
    best_distance, second_best_distance = match
    query, best, second_best = nearest_neighbors[i]
    if best_distance < ratio_threshold * second_best_distance:
        good_matches_coords.append((query, best, second_best))
    else:
        failed_matches_coords.append((query, best, second_best))

# # Plot 1: Points only
# plt.figure(figsize=(8, 8))
# for kp in keypoint_descriptors:
#     plt.scatter(kp[0], kp[1], color='orange', label='Keypoint Descriptors' if 'Keypoint Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")
# for query in query_descriptors:
#     plt.scatter(query[0], query[1], color='blue', label='Query Descriptors' if 'Query Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")
# plt.title("Points Only")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('lowes_points.png')

# Plot 2: Approved matches
plt.figure(figsize=(8, 8))
for kp in keypoint_descriptors:
    plt.scatter(kp[0], kp[1], color='orange', label='Keypoint Descriptors' if 'Keypoint Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")
for query in query_descriptors:
    plt.scatter(query[0], query[1], color='blue', label='Query Descriptors' if 'Query Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")

for query, best, second_best in good_matches_coords:
    plt.plot([query[0], best[0]], [query[1], best[1]], color='green', linestyle='-', label='Good Match' if 'Good Match' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot([query[0], second_best[0]], [query[1], second_best[1]], color='green', linestyle='--', label='Second Best (Good)' if 'Second Best (Good)' not in plt.gca().get_legend_handles_labels()[1] else "")
for query, best, second_best in failed_matches_coords:
    plt.plot([query[0], best[0]], [query[1], best[1]], color='red', linestyle='-', label='Failed Match' if 'Failed Match' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.plot([query[0], second_best[0]], [query[1], second_best[1]], color='red', linestyle='--', label='Second Best (Failed)' if 'Second Best (Failed)' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Failed and Approved Matches")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True)
plt.savefig('lowes_points_matches.png')
plt.show()


# # Plot 3: Failed matches
# plt.figure(figsize=(8, 8))
# for kp in keypoint_descriptors:
#     plt.scatter(kp[0], kp[1], color='orange', label='Keypoint Descriptors' if 'Keypoint Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")
# for query in query_descriptors:
#     plt.scatter(query[0], query[1], color='blue', label='Query Descriptors' if 'Query Descriptors' not in plt.gca().get_legend_handles_labels()[1] else "")
# for query, best, second_best in failed_matches_coords:
#     plt.plot([query[0], best[0]], [query[1], best[1]], color='red', linestyle='-', label='Failed Match' if 'Failed Match' not in plt.gca().get_legend_handles_labels()[1] else "")
#     plt.plot([query[0], second_best[0]], [query[1], second_best[1]], color='red', linestyle='--', label='Second Best (Failed)' if 'Second Best (Failed)' not in plt.gca().get_legend_handles_labels()[1] else "")
# plt.title("Failed Matches")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('lowes_points_fails.png')

