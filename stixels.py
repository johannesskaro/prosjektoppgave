import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut
import cv2

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)
    rectangular_stixel_list = []
    fused_stixel_depth_list = []

    def __init__(self, num_of_stixels = 96) -> None:
        self.num_stixels = num_of_stixels # 958//47 #gir 20I wan 

    def get_stixel_2d_points_N_frames(self) -> np.array:
        return self.stixel_2d_points_N_frames

    def add_stixel_2d_points(self, stixel_2d_points: np.array) -> None:
        self.stixel_2d_points_N_frames.append(stixel_2d_points)

    def get_stixel_width(self, img_width) -> int:
        self.stixel_width = int(img_width // self.num_stixels)
        return self.stixel_width

    def get_free_space_boundary(self, water_mask: np.array) -> np.array:

        height, width = water_mask.shape

        free_space_boundary_mask = np.zeros_like(water_mask)
        #free_space_boundary = np.zeros(width)
        free_space_boundary = np.ones(width) * height

        for j in range(width):
            for i in reversed(range(height-50)):
                if water_mask[i, j] == 0:
                    free_space_boundary_mask[i:i, j] = 1
                    free_space_boundary[j] = i
                    break

        return free_space_boundary, free_space_boundary_mask

    def get_stixels_base(self, water_mask: np.array) -> np.array:

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
    
    def create_rectangular_stixels(self, water_mask, disparity_map, depth_map):
        free_space_boundary, _ = self.get_free_space_boundary(water_mask)
        stixel_width = self.get_stixel_width(water_mask.shape[1])

        std_dev_threshold = 0.35
        #median_disp_change_threshold = 0.1
        window_size = 10
        min_stixel_height = 20

        rectangular_stixel_mask = np.zeros_like(water_mask)
        self.rectangular_stixel_list = []

        for n in range(self.num_stixels):

            stixel_range = slice(n * stixel_width, (n + 1) * stixel_width)
            stixel_base = free_space_boundary[stixel_range]
            stixel_base_height = int(np.median(stixel_base))
            stixel_top_height = stixel_base_height - min_stixel_height

            std_dev = 0
            median_row_disp_list = []

            #previous_median = np.nanmedian(disparity_map[stixel_base_height - window_size:stixel_base_height, stixel_range])
            

            for v in range(stixel_base_height, 0, -1):
                #window_range = slice(max(v - window_size, 0), v)
                #median_disparity = np.nanmedian(disparity_map[window_range, stixel_range])

                # Check if the deviation exceeds the threshold
                #if np.abs(median_disparity - previous_median) > median_disp_change_threshold:
                #    stixel_top_height = v  # Set the top height to the current row
                #    if stixel_base_height - v < min_stixel_height:
                #        stixel_top_height = stixel_base_height - min_stixel_height
                #    break

                #previous_median = median_disparity

                median_row_disp = np.nanmedian(disparity_map[v, stixel_range])
                median_row_disp_list.append(median_row_disp)
                std_dev = np.std(median_row_disp_list)
                #print(f"std_dev: {std_dev}")
                if std_dev > std_dev_threshold:
                    stixel_top_height = v
                    if stixel_base_height - v < min_stixel_height:
                        stixel_top_height = stixel_base_height - min_stixel_height
                    break

            stixel_median_disp = np.nanmedian(disparity_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel_median_depth = np.nanmedian(depth_map[stixel_top_height:stixel_base_height, stixel_range])
            stixel = [stixel_top_height, stixel_base_height, stixel_median_disp, stixel_median_depth]
            self.rectangular_stixel_list.append(stixel)
            rectangular_stixel_mask[stixel_top_height:stixel_base_height, stixel_range] = 1

        return self.rectangular_stixel_list, rectangular_stixel_mask
    
    def get_stixel_3d_points(self, camera_params):
        stixel_list = self.rectangular_stixel_list
        stixel_3d_points = np.zeros((self.num_stixels, 4, 3))
        stixel_image_points = np.zeros((4, 2))
        for n, stixel in enumerate(stixel_list):
            top_height = stixel[0]
            base_height = stixel[1]
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width
            stixel_depth = self.fused_stixel_depth_list[n] # stixel[3]

            stixel_image_points[0] = [left_bound, top_height]
            stixel_image_points[1] = [right_bound, top_height]
            stixel_image_points[2] = [right_bound, base_height]
            stixel_image_points[3] = [left_bound, base_height]
            
            depth = np.array([stixel_depth, stixel_depth, stixel_depth, stixel_depth])

            stixel_3d_points[n] = ut.calculate_3d_points(stixel_image_points[:, 0], stixel_image_points[:, 1], depth, camera_params)

        return stixel_3d_points

            
    
    def filter_lidar_points_by_stixels(self, lidar_image_points, lidar_3d_points):

        filtered_image_points = []
        filtered_3d_points = []
        stixel_indices = []
        stixel_list = self.rectangular_stixel_list

        for n, stixel in enumerate(stixel_list):
            top_height = stixel[0]
            base_height = stixel[1]
            left_bound = n * self.stixel_width
            right_bound = (n + 1) * self.stixel_width

            # Check if points fall within this stixel's bounds (horizontal and vertical in pixel coordinates)
            mask = (
                (lidar_image_points[:, 1] >= top_height) &
                (lidar_image_points[:, 1] <= base_height) &
                (lidar_image_points[:, 0] >= left_bound) &
                (lidar_image_points[:, 0] <= right_bound)
            )

            stixel_points_2d = lidar_image_points[mask]
            stixel_points_3d = lidar_3d_points[mask]  # Filter corresponding 3D points

            filtered_image_points.extend(stixel_points_2d)
            filtered_3d_points.extend(stixel_points_3d)
            stixel_indices.extend([n] * len(stixel_points_2d))

        filtered_image_points = np.array(filtered_image_points)
        filtered_3d_points = np.array(filtered_3d_points)
        stixel_indices = np.array(stixel_indices)

        return filtered_image_points, filtered_3d_points, stixel_indices
    

    def get_stixel_depth_from_lidar_points(self, lidar_3d_points, stixel_indices):
        stixel_depths = []
        for n in range(self.num_stixels):
            mask = stixel_indices == n
            stixel_points = lidar_3d_points[mask]
            
            if len(stixel_points) > 0:
                distances = stixel_points[:, 2]
                #distances = np.linalg.norm(stixel_points, axis=1)
                stixel_depth = np.nanmedian(distances)  # Take the median distance for robustness
            else:
                stixel_depth = np.nan  # Assign NaN or a placeholder for empty stixels
            stixel_depths.append(stixel_depth)
        return np.array(stixel_depths)
    
    
    def calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_map, cam_params):
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
    
    def get_polygon_points_from_lidar_and_stereo_depth(self, lidar_stixel_depths, stixel_positions, cam_params, sigma_px=1, sigma_z_lidar=0.1):

        Z = np.full(len(self.rectangular_stixel_list), np.nan)
        Z_invalid = np.array([], dtype=int)

        for n, stixel in enumerate(self.rectangular_stixel_list):
            px = stixel[2]
            z_stereo = stixel[3]
            z_lidar = lidar_stixel_depths[n]
            #print(z_lidar)

            if np.isnan(z_stereo) and np.isnan(z_lidar):
                #print(1)
                Z_invalid = np.append(Z_invalid, n)

            elif np.isnan(z_stereo) or px == 0:
                z_fused = z_lidar
                Z[n] = z_fused
                #print(2)
            elif np.isnan(z_lidar) or z_lidar == 0:
                z_fused = z_stereo
                #print(3)
                Z[n] = z_fused
            else:
                sigma_z_stereo = sigma_px * z_stereo / px
                sigma_z_squared = 1 / (1 / sigma_z_stereo**2 + 1 / sigma_z_lidar**2)  # Combine stereo and lidar depth uncertainties
                z_fused = sigma_z_squared * (z_lidar / sigma_z_lidar**2 + px / sigma_z_stereo**2)
                #print(4)
                Z[n] = z_fused
            
            #print(f"z_fused: {z_fused}")
            
        self.fused_stixel_depth_list = Z

        X = stixel_positions[:, 0]
        Y = stixel_positions[:, 1]
        X = np.delete(X, Z_invalid)
        Y = np.delete(Y, Z_invalid)
        Z = np.delete(Z, Z_invalid)
        
        points_3d = ut.calculate_3d_points(X, Y, Z, cam_params)
        points_2d = points_3d[:, [0, 2]]

        # Compute the angle of each point relative to the origin (0,0)
        angles = np.arctan2(points_2d[:, 1], points_2d[:, 0])

        # Sort points by angle to create a continuous polygon boundary
        sorted_indices = np.argsort(angles)
        points_2d_sorted = points_2d[sorted_indices]

        return points_2d_sorted
    

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
                green_stixel = np.full((stixel_base - stixel_top, stixel_width, 3), (0, 80, 0), dtype=np.uint8) #(0, 50, 0)

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
    #sorted_indices = points[:, 0].argsort()
    #points = points[sorted_indices]
    origin = np.array([0, 0])
    polygon_points = np.vstack([origin, points])

    return Polygon(polygon_points)