import numpy as np
from shapely.geometry import Polygon
from collections import deque

class Stixels:

    N = 10
    stixel_2d_points_N_frames = deque(maxlen=N)

    def __init__(self, num_of_stixels = 60) -> None:
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

    def get_stixels(self, water_mask: np.array) -> np.array:
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
    
    
def create_polygon_from_2d_points(points: list) -> Polygon:

    if len(points) < 2:
        print("Cannot create a polygon with less than 2 points.")
        return Polygon()
    #polygon_points = points[:, [0, 2]]
    polygon_points = points
    origin = np.array([0, 0])
    polygon_points = np.vstack([polygon_points, origin])

    return Polygon(polygon_points)
    
