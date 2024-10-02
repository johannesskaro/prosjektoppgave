import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
)
import json
from pointcloud import PointCloud

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

        valid = True

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
            valid = False
            return np.zeros((H, W)), np.array([0, 1, 0, 1]), valid

        plane_model, _ = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
            probability=self.probability,
        )

        if not plane_model.any():  # if plane_model is empty
            print("No plane found")
            valid = False
            return np.zeros((H, W)), np.array([0, 1, 0, 1]), valid

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
        return mask, plane_model, valid
    
    def get_segmentation_mask_from_plane_model(self, points_3d, plane_model):

        normal = plane_model[:3]
        d = plane_model[3]
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length

        mask = self.get_water_mask_from_plane_model(points_3d, plane_model)
        if self.prev_planemodel is not None:
            prev_valid = self.validity_check(
                self.prev_height, self.prev_unitnormal, height, unit_normal
            )
            init_valid = self.validity_check(
                self.init_height, self.init_unitnormal, height, unit_normal
            )

            if prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.prev_planemodel
                )

            elif not prev_valid and not init_valid:
                # if not prev_valid and not init_valid:
                mask = self.get_water_mask_from_plane_model(
                    points_3d, self.init_planemodel
                )

        return mask

    def get_image_mask(xyz, cam_params, shape):
        cx, cy = cam_params["cx"], cam_params["cy"]
        fx, fy = cam_params["fx"], cam_params["fy"]
        X_o, Y_o, Z_o = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        x = (fx * X_o / Z_o) + cx
        y = (fy * Y_o / Z_o) + cy

        mask = np.zeros(shape)
        mask[y.astype(int), x.astype(int)] = 1
        return mask

    def validity_check(self, prev_height, prev_normal, current_height, current_normal):
        if abs(prev_height - current_height) > self.validity_height_thr:
            return False
        if np.dot(prev_normal, current_normal) < np.cos(self.validity_angle_thr):
            return False
        return True

    def get_pitch(self, normal_vec=None):
        if normal_vec is None:
            normal_vec = self.prev_unitnormal
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        [a, b, c] = normal_vec
        pitch = np.arctan2(c, b)
        return pitch

    def get_roll(self, normal_vec=None):
        if normal_vec is None:
            normal_vec = self.prev_unitnormal
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        [a, b, c] = normal_vec
        roll = np.arctan2(a, b)
        return roll
    
    def get_water_mask_from_plane_model(self, points_3d, plane_model):
        normal = plane_model[:3]
        d = plane_model[3]
        H, W = self.shape
        normal_length = np.linalg.norm(normal)
        unit_normal = normal / normal_length
        height = d / normal_length
        distances = unit_normal.dot(points_3d.T) + height
        # distances = normal.dot(points_3d.T) + d
        inlier_indices_1d = np.where(np.abs(distances) < self.distance_threshold)[0]
        inlier_indices = np.unravel_index(inlier_indices_1d, (H, W))
        mask = np.zeros((H, W))
        mask[inlier_indices] = 1
        return mask

    def get_plane_model(self):
        return self.prev_planemodel
    
    def get_horizon(self, normal_vec=None):
        if normal_vec is None and self.prev_unitnormal is not None:
            normal_vec = self.prev_unitnormal
        elif normal_vec is None and self.prev_unitnormal is None:
            print("No plane parameters.")
            return None, None

        [a, b, c] = normal_vec
        fy = self.cam_params["fy"]
        cx = self.cam_params["cx"]
        cy = self.cam_params["cy"]

        x0 = 0
        xW = self.shape[1]

        k = a * cx + b * cy - c * fy
        y0 = (1 / b) * (k - a * x0)
        yW = (1 / b) * (k - a * xW)

        p1 = np.array([x0, y0])
        p2 = np.array([xW, yW])

        horizon_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        horizon_intercept = int(p1[1] - horizon_slope * p1[0])

        horizon_point0 = np.array([x0, horizon_slope * x0 + horizon_intercept]).astype(
            int
        )
        horizon_pointW = np.array([xW, horizon_slope * xW + horizon_intercept]).astype(
            int
        )
        horizon_cutoff = min(horizon_point0[1], horizon_pointW[1]) - 50

        return horizon_point0, horizon_pointW, horizon_cutoff

    

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