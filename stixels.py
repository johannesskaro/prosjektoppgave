import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)

    def __init__(self, num_of_stixels = 48) -> None:
        self.num_stixels = num_of_stixels # 958//47 #gir 20I wan 

    def get_stixel_2d_points_N_frames(self) -> np.array:
        return self.stixel_2d_points_N_frames

    def add_stixel_2d_points(self, stixel_2d_points: np.array) -> None:
        self.stixel_2d_points_N_frames.append(stixel_2d_points)

    def get_stixel_width(self, img_width) -> int:
        stixel_width = int(img_width // self.num_stixels)
        return stixel_width

    def get_free_space_boundary(self, water_mask: np.array) -> np.array:
        """
        Get the free space boundary from the water mask.

        Parameters:
        - water_mask (np.array): Water mask.

        Returns:
        - np.array: Free space boundary.
        """

        height, width = water_mask.shape

        free_space_boundary_mask = np.zeros_like(water_mask)
        #free_space_boundary = np.zeros(width)
        free_space_boundary = np.ones(width) * height

        for j in range(width):
            for i in reversed(range(height-50)):
                if water_mask[i, j] == 0:
                    free_space_boundary_mask[i, j] = 1
                    free_space_boundary[j] = i
                    break

        return free_space_boundary, free_space_boundary_mask

    def get_stixels_base(self, water_mask: np.array) -> np.array:
        """
        Obtain stixels for the input image.

        Parameters:
        - img (np.array): Input image.

        Returns:
        - np.array: Stixels result.
        """
        stixel_positions = np.zeros((self.num_stixels, 2))

        free_space_boundary, free_space_boundary_mask = self.get_free_space_boundary(water_mask)
        height, width = water_mask.shape
        stixel_width = int(width // self.num_stixels)
        stixel_mask = np.zeros_like(water_mask)


        for n in range(self.num_stixels):

            stixel = free_space_boundary[n * stixel_width:(n + 1) * stixel_width]
            stixel_y_pos = int(np.mean(stixel))
            stixel_x_pos = n * stixel_width + stixel_width // 2
            stixel_pos = np.array([stixel_x_pos, stixel_y_pos])
            #stixel_positions = np.vstack([stixel_positions, stixel_pos])
            stixel_positions[n] = stixel_pos
            stixel_mask[stixel_y_pos, n * stixel_width:(n + 1) * stixel_width] = 1
            #stixel_mask[stixel_y_pos, stixel_x_pos] = 1

        #print(stixel_positions)
        return stixel_mask, stixel_positions
    
    def create_rectangular_stixels(self, water_mask, disparity_map):
        free_space_boundary, _ = self.get_free_space_boundary(water_mask)
        stixel_width = self.get_stixel_width(water_mask.shape[1])
        #_, stixel_positions = self.get_stixels_base(water_mask)
        std_dev_threshold = 0.4

        rectangular_stixel_mask = np.zeros_like(water_mask)
        rectangular_stixel_list = []

        for n in range(self.num_stixels):
            stixel_base = free_space_boundary[n * stixel_width:(n + 1) * stixel_width]
            stixel_base_height = int(np.median(stixel_base))
            stixel_top_height = water_mask.shape[0]
            std_dev = 0
            median_row_disp_list = []
            for v in range(stixel_base_height, 0, -1):
                median_row_disp = np.median(disparity_map[v, n * stixel_width:(n + 1) * stixel_width])
                median_row_disp_list.append(median_row_disp)
                std_dev = np.std(median_row_disp_list)
                #print(f"std_dev: {std_dev}")
                if std_dev > std_dev_threshold:
                    stixel_top_height = v
                    break
            stixel_median_disp = np.median(disparity_map[stixel_top_height:stixel_base_height, n * stixel_width:(n + 1) * stixel_width])
            stixel = [stixel_top_height, stixel_base_height, stixel_median_disp]
            rectangular_stixel_list.append(stixel)
            rectangular_stixel_mask[stixel_top_height:stixel_base_height, n * stixel_width:(n + 1) * stixel_width] = 1

        return rectangular_stixel_mask, rectangular_stixel_list
    
    def filter_lidar_points_by_stixels(self, lidar_points, stixel_list, stixel_width):
        filtered_points = []
        stixel_indices = []

        for n, stixel in enumerate(stixel_list):
            top_height = stixel[0]
            base_height = stixel[1]
            left_bound = n * stixel_width
            right_bound = (n + 1) * stixel_width

            # Check if points fall within this stixel's bounds (horizontal and vertical in pixel coordinates)
            mask = (
                (lidar_points[:, 1] >= top_height) &
                (lidar_points[:, 1] <= base_height) &
                (lidar_points[:, 0] >= left_bound) &
                (lidar_points[:, 0] <= right_bound)
            )

            # Filter points and record the stixel index
            stixel_points = lidar_points[mask]

            filtered_points.extend(stixel_points)
            stixel_indices.extend([n] * len(stixel_points))


        filtered_points = np.array(filtered_points)
        stixel_indices = np.array(stixel_indices)

        return filtered_points, stixel_indices
    
        
    
    def merge_stixels_onto_image(self, stixel_list, image):

        overlay = np.zeros_like(image)
        stixel_width = self.get_stixel_width(image.shape[1])
        disp_values = [stixel[2] for stixel in stixel_list]
        min_disp = 0 #np.min(disp_values)
        max_disp = 10 #np.max(disp_values)
        

        for n, stixel in enumerate(stixel_list):
            stixel_top = stixel[0]
            stixel_base = stixel[1]
            stixel_disp = stixel[2]

            if stixel_base > stixel_top and stixel_width > 0:

                #normalized_disp = np.uint8(255 * (stixel_disp - min_disp) / (max_disp - min_disp))
                #normalized_disp_array = np.full((stixel_base - stixel_top, stixel_width), normalized_disp, dtype=np.uint8)
                #colored_stixel = cv2.applyColorMap(normalized_disp_array, cv2.COLORMAP_JET)
                green_stixel = np.full((stixel_base - stixel_top, stixel_width, 3), (0, 50, 0), dtype=np.uint8)

                overlay[stixel_top:stixel_base, n * stixel_width:(n + 1) * stixel_width] = green_stixel

                # Add a border (rectangle) around the stixel
                cv2.rectangle(overlay, 
                        (n * stixel_width, stixel_top),  # Top-left corner
                        ((n + 1) * stixel_width, stixel_base),  # Bottom-right corner
                        (0,0,0),  # Color of the border (BGR)
                        2)  # Thickness of the border

        alpha = 0.8  # Weight of the original image
        beta = 1  # Weight of the overlay
        gamma = 0.0  # Scalar added to each sum

        blended_image = cv2.addWeighted(image, alpha, overlay, beta, gamma)
        return blended_image
        

    
def create_polygon_from_2d_points(points: list) -> Polygon:

    if len(points) < 2:
        print("Cannot create a polygon with less than 2 points.")
        return Polygon()
    #polygon_points = points[:, [0, 2]]
    polygon_points = points
    origin = np.array([0, 0])
    polygon_points = np.vstack([polygon_points, origin])

    return Polygon(polygon_points)
    
def calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_map, cam_params):
    height, width = depth_map.shape
    # add stixels in front left and right

    stixel_positions = stixel_positions[stixel_positions[:, 0].argsort()]

    d = np.array([])
    d_invalid = np.array([])
    for n, stixel_pos in enumerate(stixel_positions):
        x_start = max(0, n * stixel_width - stixel_width // 2)
        x_end = min(depth_map.shape[1], (n + 1) * stixel_width - stixel_width // 2)
        depth_along_stixel = depth_map[int(stixel_pos[1]), x_start:x_end]
        depth_along_stixel = depth_along_stixel[depth_along_stixel > 0]
        depth_along_stixel = depth_along_stixel[~np.isnan(depth_along_stixel)]
        if depth_along_stixel.size == 0: #no depth values in stixel
            #stixel_positions = np.delete(stixel_positions, n, axis=0)
            d_invalid = np.append(d_invalid, int(n))
        else:
            median_depth = np.median(depth_along_stixel[depth_along_stixel > 0])
            #avg_depth = np.mean(depth_along_stixel[depth_along_stixel > 0])
            d = np.append(d, median_depth)
    
    d_invalid = np.array(d_invalid, dtype=int)
    X = stixel_positions[:, 0]
    X = np.delete(X, d_invalid)
    Y = stixel_positions[:, 1]
    Y = np.delete(Y, d_invalid)
    points_3d = ut.calculate_3d_points(X, Y, d, cam_params)

    points_2d = points_3d[:, [0, 2]]

    return points_2d
