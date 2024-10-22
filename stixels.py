import numpy as np
from shapely.geometry import Polygon
from collections import deque
import utilities as ut

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)

    def __init__(self, num_of_stixels = 30) -> None:
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
        free_space_boundary = np.zeros(width)

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
        std_dev_threshold = 1

        rectangular_stixel_mask = np.zeros_like(water_mask)

        for n in range(self.num_stixels):
            stixel_base = free_space_boundary[n * stixel_width:(n + 1) * stixel_width]
            stixel_base_height = int(np.mean(stixel_base))
            stixel_top_height = water_mask.shape[0]
            std_dev = 0
            median_disp_list = []
            for v in range(stixel_base_height, 0, -1):
                median_disp = np.median(disparity_map[v, n * stixel_width:(n + 1) * stixel_width])
                median_disp_list.append(median_disp)
                std_dev = np.std(median_disp_list)
                #print(f"std_dev: {std_dev}")
                if std_dev > std_dev_threshold:
                    stixel_top_height = v
                    break
            rectangular_stixel_mask[stixel_top_height:stixel_base_height, n * stixel_width:(n + 1) * stixel_width] = 1

        return rectangular_stixel_mask

    
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
