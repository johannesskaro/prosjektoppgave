import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
)
import json

class RWPS:
    def __init__(self, config_file=None) -> None:
        if config_file is not None:
            # Load configuration file
            self.set_config(config_file)

        else:
            # Use default configuration
            self.config_file = None
            self.distance_threshold = 0.01
            self.ransac_n = 3
            self.num_iterations = 1000
            self.probability = 0.99999999

            self.validity_height_thr = 0.1  # m
            self.validity_angle_thr = 5  # deg
            self.validity_min_inliers = 100

        self.prev_planemodel = None
        self.prev_planemodel_disp = None
        self.prev_height = 0
        self.prev_unitnormal = np.array([0, 1, 0])
        self.prev_mask = None
        self.prev_residual_threshold = None
        self.prev_mask_ds = None
        self.invalid = None
        self.counter = 0
        self.sigma_e = 1 / 2

    def set_invalid(self, p1, p2, shape):
        self.invalid = invalid_mask(p1, p2, shape)
        return self.invalid
    
    def set_config(self, config_file):
        self.set_config_xyz(config_file)

    def set_config_xyz(self, config_file):
        """
        Set parameters for RANSAC plane segmentation using 3D point cloud
        """
        self.config_file = config_file
        config_data = json.load(open(config_file))
        self.distance_threshold = config_data["RANSAC"]["distance_threshold"]
        self.ransac_n = config_data["RANSAC"]["ransac_n"]
        self.num_iterations = config_data["RANSAC"]["num_iterations"]
        self.probability = config_data["RANSAC"]["probability"]
        self.validity_height_thr = config_data["plane_validation"]["height_thr"]
        self.validity_angle_thr = config_data["plane_validation"]["angle_thr"]
        self.validity_min_inliers = config_data["plane_validation"]["min_inliers"]
        self.initial_roll = None
        self.initial_pitch = None
        self.disp_deviations_inliers_array = {}
        self.counter = 0

    def set_initial_pitch(self, pitch):
        self.initial_pitch = pitch

    def set_initial_roll(self, roll):
        self.initial_roll = roll

    def set_camera_params(self, cam_params, P1, camera_height=None):
        self.cam_params = cam_params
        self.P1 = P1
        self.camera_height = camera_height

    def segment_water_plane_using_point_cloud(
        self,
        img: np.array,
        depth: np.array,
    ) -> np.array:

        assert self.cam_params is not None, "Camera parameters are not provided."

        if self.config_file is None:
            print(
                "Warning: Configuration file is not provided. Using default parameters."
            )

        (H, W, _) = img.shape
        self.shape = (H, W)

        pcd = PointCloud()
        pcd = pcd.create_from_img_and_depth(img, depth, self.cam_params).get_o3d_pc()
        points_3d = np.asarray(pcd.points)

        # Initialization
        if self.prev_planemodel is None:
            inlier_mask = np.zeros((H, W))
            inlier_mask[H // 2 :, :] = 1

        else:
            ### Use previous inliers
            inlier_mask = self.prev_mask
            ### Horizon line
            p1, p2, _ = self.get_horizon()  # self.mean_point)
            x_coords, y_coords = np.meshgrid(
                np.arange(inlier_mask.shape[1]), np.arange(inlier_mask.shape[0])
            )
            line_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            line_intercept = p1[1] - line_slope * p1[0]
            line_mask = y_coords > (line_slope * x_coords + line_intercept)
            inlier_mask = np.logical_and(inlier_mask, line_mask)

        pcd = pcd.select_by_index(np.where(inlier_mask.flatten() == 1)[0])

        # Check if there are enough points to segment plane
        if len(pcd.points) < self.ransac_n:
            print("Not enough points to segment plane")
            self.prev_planemodel = None
            return np.zeros((H, W)), np.array([0, 1, 0, 1])

        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
            probability=self.probability,
        )

        if not plane_model.any():  # if plane_model is empty
            print("No plane found")
            return np.zeros((H, W)), np.array([0, 1, 0, 1])

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length  # abs(d) #/ normal_length

        # if self.prev_planemodel is not np.array([0,1,0]):
        if self.prev_planemodel is not None:
            self.init_planemodel = plane_model
            self.init_height = height
            self.init_unitnormal = unit_normal

        mask = self.get_segmentation_mask_from_plane_model(points_3d, plane_model)

        self.prev_planemodel = plane_model
        self.prev_height = height
        self.prev_unitnormal = unit_normal
        self.prev_mask = mask
        return mask, plane_model



    



def invalid_mask(p1, p2, shape):
    # Define the points
    H, W = shape
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the slope (m) and intercept (b) of the line
    if x2 - x1 == 0:
        m = 99999999999999
        b = x1
    else:
        m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Create the invalid_mask
    invalid_mask = np.zeros((H, W), dtype=bool)

    # Generate a grid of coordinates
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Calculate the y values of the line at each x coordinate
    y_line = m * x_coords + b

    # Mark everything under the line as True
    invalid_mask = y_coords >= y_line

    return invalid_mask