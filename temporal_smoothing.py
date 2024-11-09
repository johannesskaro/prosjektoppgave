import numpy as np
from collections import deque
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class TemporalSmoothing:

    def __init__(self, N, camera_matrix, R_imu_to_camera=None, t_imu_to_camera=None) -> None:
        self.N = N  # Number of past frames to consider
        self.camera_matrix = camera_matrix  # Camera intrinsic matrix
        #self.R_imu_to_camera = R_imu_to_camera  # Rotation from IMU to camera frame
        #self.t_imu_to_camera = t_imu_to_camera.reshape((3, 1))  # Translation from IMU to camera fr
        
        #self.past_N_masks = deque(maxlen=N)
        #self.past_N_orientations = deque(maxlen=N)

            # Initialize lists to store past masks and orientations
        self.past_N_masks = []
        self.past_N_orientations = []

    def add_mask(self, mask):
        # Keep only the last N masks
        self.past_N_masks.append(mask)
        if len(self.past_N_masks) > self.N:
            self.past_N_masks.pop(0)

    def add_orientation(self, roll, pitch, yaw):
        # Keep only the last N orientations
        self.past_N_orientations.append((roll, pitch, yaw))
        if len(self.past_N_orientations) > self.N:
            self.past_N_orientations.pop(0)

    def get_smoothed_water_mask(self, water_mask: np.array) -> np.array:

        mask_sum = np.sum(self.past_N_masks, axis=0)

        smoothed_water_mask = np.zeros_like(water_mask)
        threshold = self.N // 2
        thresholded_mask = (mask_sum > threshold).astype(int)

        smoothed_water_mask = np.logical_or(water_mask, thresholded_mask).astype(int)

        self.add_mask(water_mask)

        return smoothed_water_mask


    def get_smoothed_ego_motion_compensated_mask(self, mask_curr, roll_curr, pitch_curr, yaw_curr):

        # Update past masks and orientations
        # Get compensated masks
        past_N_compensated_masks = self.get_ego_motion_compensated_masks(roll_curr, pitch_curr, yaw_curr)
        
        # Combine masks using majority voting
        mask_sum = np.sum(past_N_compensated_masks, axis=0)
        threshold = self.N * 2 // 3
        thresholded_mask = (mask_sum > threshold).astype(np.uint8)
        smoothed_water_mask = np.logical_or(mask_curr, thresholded_mask).astype(np.uint8)

        self.add_mask(mask_curr)
        self.add_orientation(roll_curr, pitch_curr, yaw_curr)
        
        return smoothed_water_mask

    def get_ego_motion_compensated_masks(self, roll_curr, pitch_curr, yaw_curr):
        past_N_compensated_masks = []
        R_curr = self.get_rotation_matrix(roll_curr, pitch_curr, yaw_curr)
        
        for i, (mask_prev, (roll_prev, pitch_prev, yaw_prev)) in enumerate(zip(self.past_N_masks, self.past_N_orientations)):
            R_prev = self.get_rotation_matrix(roll_prev, pitch_prev, yaw_prev)
            
            # Compute the relative rotation in the IMU frame
            R_rel_imu = R_curr @ R_prev.T
            
            # Compute the induced translation in the IMU frame
            t_induced_imu = R_rel_imu @ self.t_imu_to_camera - self.t_imu_to_camera
            
            # Transform rotation and translation to the camera frame
            R_rel_camera = self.R_imu_to_camera @ R_rel_imu @ self.R_imu_to_camera.T
            t_rel_camera = self.R_imu_to_camera @ t_induced_imu
            
            # Define plane normal and distance (assuming plane perpendicular to Z-axis)
            n = np.array([[0], [0], [1]])  # Normal vector in camera frame
            d = 0.7  # Approximate distance to the plane (adjust as needed)
            
            # Compute homography
            H = self.compute_homography(R_rel_camera, t_rel_camera, n, d, self.camera_matrix)
            
            # Warp the previous mask
            warped_mask = cv2.warpPerspective(mask_prev.astype(np.uint8), H, (mask_prev.shape[1], mask_prev.shape[0]), flags=cv2.INTER_NEAREST)
            past_N_compensated_masks.append(warped_mask)
        
        # Include the current mask as well
        past_N_compensated_masks = np.array(past_N_compensated_masks)
        return past_N_compensated_masks
    
    def compute_homography(self, R, t, n, d, K):
        """
        Computes the homography matrix H = K * (R - (t * n^T) / d) * K_inv
        """
        K_inv = np.linalg.inv(K)
        Rt = R - (t @ n.T) / d
        H = K @ Rt @ K_inv
        return H

    def get_rotation_matrix(self, roll, pitch, yaw):
        # Convert angles to radians if necessary
        angles = np.array([yaw, pitch, roll])  # Order: yaw, pitch, roll
        # Create rotation object using ZYX sequence for NED convention
        rotation = R.from_euler('ZYX', angles, degrees=False)
        # If needed, adjust for axis directions (e.g., invert Z-axis)
        rotation_matrix = rotation.as_matrix()
        # Invert Z-axis if IMU's Z-axis points down
        #invert_z = np.diag([1, 1, -1])
        #rotation_matrix = invert_z @ rotation_matrix @ invert_z
        return rotation_matrix
    
    def ned_to_enu_rotation(self, R_ned):

        T_ned_to_enu = np.array([[0, 1, 0], 
                                [1, 0, 0], 
                                [0, 0, -1]])
        
        # Convert the rotation matrix from NED to ENU
        R_enu = T_ned_to_enu @ R_ned @ T_ned_to_enu.T
        
        return R_enu

    def ned_to_enu_vector(self, v_ned):
        T_ned_to_enu = np.array([[0, 1, 0], 
                                [1, 0, 0], 
                                [0, 0, -1]])
        
        # Convert the vector from NED to ENU
        v_enu = T_ned_to_enu @ v_ned
        
        return v_enu
    
    #def add_orientation(self, roll, pitch, yaw):
    #    self.past_N_orientations.append((roll, pitch, yaw))
    
    #def add_mask(self, mask):     
    #    self.past_N_masks.append(mask)

    # Function to convert roll, pitch, yaw to rotation matrix
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # Rotation matrix for yaw (Z axis)
        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        # Rotation matrix for pitch (Y axis)
        R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

        # Rotation matrix for roll (X axis)
        R_roll = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        # Full rotation matrix (Yaw -> Pitch -> Roll)
        R = R_yaw @ R_pitch @ R_roll
        return R

    # Warp mask using relative rotation
    def warp_mask(self, mask_prev, R_rel, camera_matrix):
        # Get pixel coordinates in 2D image space
        height, width = mask_prev.shape
        y, x = np.indices((height, width))
        pixels_2d = np.stack([x, y, np.ones_like(x)], axis=-1)  # Homogeneous coords

        # Convert to 3D camera coordinates
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        pixels_3d = pixels_2d @ camera_matrix_inv.T  # (N, 3)

        # Apply relative rotation to each 3D point
        pixels_3d_rotated = pixels_3d @ R_rel.T

        # Project back to 2D image space
        pixels_2d_rotated = pixels_3d_rotated @ camera_matrix.T
        pixels_2d_rotated /= pixels_2d_rotated[..., 2:]  # Normalize by z

        # Use OpenCV's remap for warping the previous mask
        map_x, map_y = pixels_2d_rotated[..., 0], pixels_2d_rotated[..., 1]
        mask_prev = mask_prev.astype(np.uint8) * 255  # If binary mask (0 and 1), scale to 0 and 255
        warped_mask = cv2.remap(mask_prev, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC)
        warped_mask_bool = warped_mask > 0
        return warped_mask_bool
    
    def warp_mask_with_camera_offset(self, mask_prev, R_rel, t_induced, camera_matrix, y_offset=15):
        """
        Warps the previous mask using the relative rotation and camera-to-IMU offset vector.

        Parameters:
        - mask_prev: The previous mask (2D numpy array).
        - R_rel: The relative rotation matrix (3x3).
        - camera_matrix: The intrinsic camera matrix (3x3).
        - r_camera_to_imu: The vector from the IMU to the camera in the body frame (3x1 numpy array).

        Returns:
        - warped_mask: The warped mask in the current frame.
        """

        # Get pixel coordinates in 2D image space
        height, width = mask_prev.shape
        y, x = np.indices((height, width))
        pixels_2d = np.stack([x, y, np.ones_like(x)], axis=-1)  # Homogeneous coords

        # Convert to 3D camera coordinates (using inverse camera matrix)
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        pixels_3d = pixels_2d @ camera_matrix_inv.T  # Convert 2D to 3D coordinates

        # Apply the relative rotation to the 3D points
        pixels_3d_rotated = pixels_3d @ R_rel.T

        # Apply induced translation to account for the offset from the IMU
        pixels_3d_rotated += t_induced

        # Project back to 2D image space using the camera matrix
        pixels_2d_rotated = pixels_3d_rotated @ camera_matrix.T
        pixels_2d_rotated /= pixels_2d_rotated[..., 2:]  # Normalize by z

        # Introduce a bias/offset to the Y-coordinates (move the mask down)
        pixels_2d_rotated[..., 1] -= y_offset
    
        # Use OpenCV's remap for warping the previous mask
        map_x, map_y = pixels_2d_rotated[..., 0], pixels_2d_rotated[..., 1]
        mask_prev = mask_prev.astype(np.uint8) * 255  # If binary mask (0 and 1), scale to 0 and 255
        warped_mask = cv2.remap(mask_prev, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC)
        warped_mask_bool = warped_mask > 0

        return warped_mask_bool


    def get_relative_rotation(self, roll_prev, pitch_prev, yaw_prev, roll_curr, pitch_curr, yaw_curr):
        # Calculate relative rotation matrix
        R_prev = self.euler_to_rotation_matrix(roll_prev, pitch_prev, yaw_prev)
        R_curr = self.euler_to_rotation_matrix(roll_curr, pitch_curr, yaw_curr)
        R_rel = R_curr @ R_prev.T

        return R_rel
    
    def plot_N_compensated_masks(self, past_N_compensated_masks):
        masks = np.array(past_N_compensated_masks)
        N = masks.shape[0]  # Number of masks
        cols = 3  # Number of columns in the grid (you can adjust this)
        rows = (N + cols - 1) // cols  # Calculate number of rows based on N and cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Adjust the size as needed

        # Flatten axes for easy indexing
        axes = axes.flatten()

        for i in range(N):
            ax = axes[i]
            ax.imshow(masks[i], cmap='gray')  # Plot the mask in grayscale
            ax.set_title(f"Mask {i + 1}")
            ax.axis('off')  # Hide axis

        # Hide any unused axes (if N is not a perfect multiple of cols)
        for i in range(N, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_compensated_vs_noncompensated_masks(self, past_N_compensated_masks):
        """
        Plots the last N ego motion-compensated masks and compares them against the non-compensated masks.

        Parameters:
        - masks_compensated: A 3D numpy array of shape (N, height, width) containing compensated masks.
        - masks_not_compensated: A 3D numpy array of shape (N, height, width) containing non-compensated masks.
        """
        masks_compensated = np.array(past_N_compensated_masks)
        masks_not_compensated = np.array(self.past_N_masks)
        N = masks_not_compensated.shape[0]  # Number of masks
        cols = 2  # We want to plot two columns for comparison: compensated vs non-compensated
        rows = N  # One row for each mask comparison

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Adjust the size as needed

        for i in range(N):
            # Plot the non-compensated mask
            axes[i, 0].imshow(masks_not_compensated[i], cmap='gray')
            axes[i, 0].set_title(f"Non-Compensated Mask {i + 1}")
            axes[i, 0].axis('off')

            # Plot the compensated mask
            axes[i, 1].imshow(masks_compensated[i], cmap='gray')
            axes[i, 1].set_title(f"Compensated Mask {i + 1}")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_overlay_compensated_vs_noncompensated(self, past_N_compensated_masks, current_mask):
        """
        Plots an overlay of the last N ego motion-compensated masks and non-compensated masks.
        Different colors will be used to differentiate between the two.

        Parameters:
        - masks_compensated: A 3D numpy array of shape (N, height, width) containing compensated masks.
        - masks_not_compensated: A 3D numpy array of shape (N, height, width) containing non-compensated masks.
        """
        masks_compensated = np.array(past_N_compensated_masks)
        masks_not_compensated = np.array(self.past_N_masks)
        N = masks_not_compensated.shape[0]  # Number of masks
        fig, axes = plt.subplots(1, N, figsize=(N * 4, 4))  # Adjust the width by multiplying N
        
        if N == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one mask

        for i in range(N):
            ax = axes[i]

            # Create a blank RGB image for overlay
            overlay = np.zeros((*masks_compensated[i].shape, 3), dtype=np.uint8)

            # Add non-compensated mask in red (R channel)
            #overlay[masks_not_compensated[i] > 0, 0] = 255  # Red for non-compensated

            # Add compensated mask in green (G channel)
            overlay[masks_compensated[i] > 0, 1] = 255  # Green for compensated

            # Add current mask in blue (B channel)
            overlay[current_mask > 0, 2] = 255  # Blue for the current mask

            # Plot the overlay
            ax.imshow(overlay)
            ax.set_title(f"Mask {i + 1}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    def get_smoothed_ego_motion_compensated_mask_old(self, mask_curr, roll_curr, pitch_curr, yaw_curr):

        past_N_compensated_masks = self.get_ego_motion_compensated_masks_old(mask_curr, roll_curr, pitch_curr, yaw_curr)
        mask_sum = np.sum(past_N_compensated_masks, axis=0)
        smoothed_water_mask = np.zeros_like(mask_curr)
        threshold = self.N // 2
        thresholded_mask = (mask_sum > threshold).astype(int)
        smoothed_water_mask = np.logical_or(mask_curr, thresholded_mask).astype(int)

        self.add_mask(mask_curr)
        self.add_orientation(roll_curr, pitch_curr, yaw_curr)
        return smoothed_water_mask

    def get_ego_motion_compensated_masks_old(self, mask_curr, roll_curr, pitch_curr, yaw_curr):
        r_camera_to_imu = self.t_imu_to_camera
        past_N_compensated_masks = np.zeros((self.N, *mask_curr.shape))


        for i, (roll_prev, pitch_prev, yaw_prev) in enumerate(self.past_N_orientations):
            R_rel = self.get_relative_rotation(roll_prev, pitch_prev, yaw_prev, roll_curr, pitch_curr, yaw_curr)
            t_induced = np.dot(R_rel, r_camera_to_imu) - r_camera_to_imu
            warped_mask = self.warp_mask_with_camera_offset(self.past_N_masks[i], R_rel, t_induced, self.camera_matrix)
            past_N_compensated_masks[i] = warped_mask

        return past_N_compensated_masks