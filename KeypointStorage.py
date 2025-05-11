import numpy as np
from scipy.spatial import KDTree
import cv2 as cv
from shapely.geometry import Point, Polygon
from matplotlib import cm
from matplotlib.colors import Normalize

g_minumum = 0.3
g_maximum = 1.5
g_initial_reliability = 0.7

class KeypointStorage:
    def __init__(self, threshold=5):
        """
        Initializes an empty storage for keypoints.
        
        Parameters:
        - threshold: Distance threshold to consider a keypoint as an existing one (for updates).
        """
        self.new_image_polygons = []
        self.previous_boxes = []
        self.search_areas = []

        self.keypoints_coords = []       # List to store coordinates for KDTree
        self.keypoints_data = {}         # Dictionary to store keypoints data
        self.kdtree = None               # KDTree for fast spatial lookup
        self.threshold = threshold       # Distance threshold for matching existing keypoints

    def add_new_image_polygon(self, new_image_polygon):
        """
        Adds a new image polygon to the storage.
        
        Parameters:
        - new_image_polygon: List of 4 coordinates (each a tuple of x, y) representing the corners of the new image.
        """
        self.new_image_polygons.append(new_image_polygon)

    def add_previous_box(self, previous_box):
        """
        Adds a previous box to the storage.
        
        Parameters:
        - previous_box: List of 4 coordinates (each a tuple of x, y) representing the corners of the previous box.
        """
        self.previous_boxes.append(previous_box)

    def add_search_area(self, search_area):
        """
        Adds a search area to the storage.
        
        Parameters:
        - search_area: List of 4 coordinates (each a tuple of x, y) representing the corners of the search area.
        """
        self.search_areas.append(search_area)

    def _build_kdtree(self):
        """Rebuilds the k-d tree after adding new keypoints."""
        if self.keypoints_coords:
            self.kdtree = KDTree(self.keypoints_coords)
        else:
            self.kdtree = None

    def add_or_update_keypoints(self, new_keypoints, new_descriptors, iteration, reliability_multiplier=1.0, color=(255, 255, 255), thickness=2):
        """
        Adds or updates keypoints with reliability values.
        
        Parameters:
        - new_keypoints: List of new keypoints (e.g., list of cv2.KeyPoint objects).
        - new_descriptors: List of descriptors corresponding to each new keypoint.
        - iteration: Current iteration number (used for tracking).
        - reliability_multiplier: Multiplier for the reliability value of existing keypoints.
        - color: Color of the keypoints in BGR format (default is white).
        - thickness: Thickness of the keypoints (default is 2).
        """
        for kp, descriptor in zip(new_keypoints, new_descriptors):
            kp_coords = kp.pt  # Get the (x, y) coordinates of the keypoint
            found = False
            
            # Step 1: Check if this keypoint is close to an existing one
            if self.kdtree:
                indices = self.kdtree.query_ball_point(kp_coords, self.threshold)
                for idx in indices:
                    # If a keypoint at similar coordinates is found, update its reliability instead of adding a new one
                    if np.linalg.norm(np.array(kp_coords) - np.array(self.keypoints_coords[idx])) < self.threshold:
                        self.keypoints_data[idx]["reliability"] *= reliability_multiplier
                        
                        reliability = self.keypoints_data[idx]["reliability"]

                        minimum = g_minumum
                        maximum = g_maximum
                        norm = Normalize(vmin=minimum, vmax=maximum)
                        normalized_reliability = norm(reliability)  # Normalize the reliability value
                        color = cm.plasma(normalized_reliability)[:3]
                        color = tuple([int(c * 255) for c in color])
                        self.keypoints_data[idx]["color"] = color
                        
                        found = True
                        break

            # Step 2: If the keypoint is new, add it to the storage
            if not found:
                new_id = len(self.keypoints_coords)
                self.keypoints_coords.append(kp_coords)
                reliability = g_initial_reliability 
                minimum = g_minumum
                maximum = g_maximum
                # normalized_reliability = min(max(reliability / maximum, minimum), 1.0)
                norm = Normalize(vmin=minimum, vmax=maximum)
                normalized_reliability = norm(reliability)
                color = cm.plasma(normalized_reliability)[:3]
                color = tuple([int(c * 255) for c in color])
                self.keypoints_data[new_id] = {
                    "coords": kp_coords,
                    "reliability": g_initial_reliability,
                    # "descriptor": kp.descriptor if hasattr(kp, 'descriptor') else None,
                    "descriptor": descriptor,
                    "scale": kp.size,
                    "angle": kp.angle,
                    "response": kp.response,
                    "octave": kp.octave,
                    "color": color,
                    "iteration": iteration
                }

        # Step 3: Rebuild the k-d tree to include new keypoints
        self._build_kdtree()

    def query_keypoints(self, rect_points) -> list:
        """
        Retrieve keypoints within a rectangle defined by four points.
        
        Parameters:
        - rect_points: List of 4 coordinates (each a tuple of x, y) representing the corners of the rectangle.
        
        Returns:
        - A list of keypoint data dictionaries within the specified rectangle.
        """
        if not self.kdtree:
            return []

        # Create a polygon from the rectangle points
        polygon = Polygon(rect_points)

        # Retrieve full keypoint data for each index
        keypoints_within_rect = []
        descriptors = []
        for kp_data in self.keypoints_data.values():
            kp_reliability = kp_data["reliability"]

            if kp_reliability <= g_minumum:
                continue

            kp_coords = kp_data["coords"]

            point = Point(kp_coords)
            if polygon.contains(point):
                keypoints_within_rect.append(kp_data)
                descriptors.append(kp_data["descriptor"])

        return keypoints_within_rect, descriptors

    def get_all_keypoints(self) -> list:
        """
        Retrieve all stored keypoints.
        
        Returns:
        - A list of dictionaries containing all keypoint data.
        """
        return list(self.keypoints_data.values())
    
    def visualize_keypoints(self, dot_size=2):
        """
        Visualizes all keypoints in storage on a blank black canvas.

        Parameters:
        - canvas_size: Tuple specifying the (height, width) of the canvas.
        - dot_color: Color of the dots representing keypoints in BGR format (default is white).
        - dot_size: Size (thickness) of the dots for each keypoint.
        
        Returns:
        - canvas: The resulting image with keypoints drawn.
        """
        # Create a blank black canvas

        minimum = np.min(self.keypoints_coords, axis=0)
        maximum = np.max(self.keypoints_coords, axis=0)

        offset_x = 0
        if minimum[0] < 0:
            offset_x = -minimum[0]

        offset_y = 0
        if minimum[1] < 0:
            offset_y = -minimum[1]

        canvas_size = (maximum - minimum).astype(int)

        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

        # Draw each keypoint on the canvas
        for keypoint_data in self.keypoints_data.values():
            x, y = keypoint_data["coords"]
            color = keypoint_data["color"]
            reliability = keypoint_data["reliability"]
            # if 0 <= int(x) < canvas_size[0] and 0 <= int(y) < canvas_size[1]:
            # cv.circle(canvas, (int(x+offset_x), int(y+offset_y)), int(dot_size*reliability), color, -1)
            cv.circle(canvas, (int(x+offset_x), int(y+offset_y)), dot_size, color, -1)
        
        search_area = self.search_areas[-1]
        for i in range(4):
            cv.line(canvas, (int(search_area[i][0]+offset_x), int(search_area[i][1]+offset_y)), (int(search_area[(i+1)%4][0]+offset_x), int(search_area[(i+1)%4][1]+offset_y)), (255, 255, 255), 2)

        previous_box = self.previous_boxes[-1]
        for i in range(4):
            cv.line(canvas, (int(previous_box[i][0]+offset_x), int(previous_box[i][1]+offset_y)), (int(previous_box[(i+1)%4][0]+offset_x), int(previous_box[(i+1)%4][1]+offset_y)), (0, 255, 0), 2)

        new_image_polygon = self.new_image_polygons[-1]
        for i in range(4):
            cv.line(canvas, (int(new_image_polygon[i][0]+offset_x), int(new_image_polygon[i][1]+offset_y)), (int(new_image_polygon[(i+1)%4][0]+offset_x), int(new_image_polygon[(i+1)%4][1]+offset_y)), (0, 0, 255), 2)

        return canvas
