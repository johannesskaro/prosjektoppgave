import numpy as np


def get_smoothed_ego_motion_compensated_mask_old(self, mask_curr, roll_curr, pitch_curr, yaw_curr, camera_matrix, r_camera_to_imu):

    past_N_compensated_masks = self.get_ego_motion_compensated_masks(mask_curr, roll_curr, pitch_curr, yaw_curr, camera_matrix, r_camera_to_imu)
    mask_sum = np.sum(past_N_compensated_masks, axis=0)
    smoothed_water_mask = np.zeros_like(mask_curr)
    threshold = self.N // 2
    thresholded_mask = (mask_sum > threshold).astype(int)
    smoothed_water_mask = np.logical_or(mask_curr, thresholded_mask).astype(int)

    self.add_mask(mask_curr)
    self.add_orientation(roll_curr, pitch_curr, yaw_curr)
    return smoothed_water_mask

def get_ego_motion_compensated_masks_old(self, mask_curr, roll_curr, pitch_curr, yaw_curr, camera_matrix, r_camera_to_imu):

    past_N_compensated_masks = np.zeros((self.N, *mask_curr.shape))


    for i, (roll_prev, pitch_prev, yaw_prev) in enumerate(self.past_N_orientations):
        R_rel = self.get_relative_rotation(roll_prev, pitch_prev, yaw_prev, roll_curr, pitch_curr, yaw_curr)
        t_induced = np.dot(R_rel, r_camera_to_imu) - r_camera_to_imu
        warped_mask = self.warp_mask_with_camera_offset(self.past_N_masks[i], R_rel, t_induced, camera_matrix)
        past_N_compensated_masks[i] = warped_mask

    return past_N_compensated_masks