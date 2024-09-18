import cv2
import numpy as np
import scipy.io
from scipy.interpolate import interp1d


def blend_image_with_mask(img, mask, color=[0, 0, 255], alpha1=1, alpha2=1):
    # Convert binary mask to a colored mask (e.g., red)
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 1] = color

    # Blend the original image and the colored mask
    blended = cv2.addWeighted(img, alpha1, colored_mask, alpha2, 0)
    return blended


def corresponding_pixels(mask1: np.array, mask2: np.array) -> int:

    if mask1.shape != mask2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    corresponding_count = np.sum(mask1 == mask2)

    return corresponding_count


def rotate_point(x, y, image_width, image_height, roll_rad, initial_roll_rad=0):
    """
    Rotate a point (x, y) in the image by the roll angle.

    :param x: x-coordinate of the point
    :param y: y-coordinate of the point
    :param roll_angle_rad: Roll angle in radians
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: Rotated point (x', y')
    """

    # Translate point to origin-centered coordinates
    x -= image_width / 2
    y -= image_height / 2

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(roll_rad - initial_roll_rad), -np.sin(roll_rad - initial_roll_rad)],
            [np.sin(roll_rad - initial_roll_rad), np.cos(roll_rad - initial_roll_rad)],
        ]
    )

    # Rotated point
    x_rotated, y_rotated = rotation_matrix @ np.array([x, y])

    # Translate back to image coordinates
    x_rotated += image_width / 2
    y_rotated += image_height / 2

    return round(x_rotated), round(y_rotated)


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_3d_point(x_o, y_o, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]
    X_o = d * (x_o - cx) / fx
    Y_o = d * (y_o - cy) / fy
    return float(X_o), float(Y_o), d


def calculate_3d_points(X, Y, d, cam_params):
    cx, cy = cam_params["cx"], cam_params["cy"]
    fx, fy = cam_params["fx"], cam_params["fy"]

    X_o = d * (X - cx) / fx
    Y_o = d * (Y - cy) / fy
    Z_o = d

    return np.array([X_o, Y_o, Z_o]).T


def calculate_3d_points_from_mask(mask, depth_map, cam_params):
    if mask.shape != depth_map.shape:
        raise ValueError("Mask and depth map must have the same dimensions")

    Y, X = np.where(np.logical_and(mask > 0, depth_map > 0))
    d = depth_map[Y, X]

    points_3d = calculate_3d_points(X, Y, d, cam_params)

    return points_3d


from scipy.spatial.transform import Rotation


def mods_2_intrinsics_extrinsics(M1, M2, D1, D2, R, T, size):
    intrinsics = {
        "stereo_left": {
            "fx": M1[0, 0],
            "fy": M1[1, 1],
            "cx": M1[0, 2],
            "cy": M1[1, 2],
            "distortion_coefficients": D1,
            "image_width": size[0],
            "image_height": size[1],
        },
        "stereo_right": {
            "fx": M2[0, 0],
            "fy": M2[1, 1],
            "cx": M2[0, 2],
            "cy": M2[1, 2],
            "distortion_coefficients": D2,
            "image_width": size[0],
            "image_height": size[1],
        },
    }

    extrinsics = {"rotation_matrix": R, "translation": T / 1000}
    return intrinsics, extrinsics


def get_fov(fx, fy, W, H, type="rad"):
    fov_x = 2 * np.arctan(W / (2 * fx))
    fov_y = 2 * np.arctan(H / (2 * fy))
    if type == "deg":
        fov_x = np.rad2deg(fov_x)
        fov_y = np.rad2deg(fov_y)
    return fov_x, fov_y


def invert_transformation(H):
    R = H[:3, :3]
    T = H[:3, 3]
    H_transformed = np.block(
        [
            [R.T, -R.T.dot(T)[:, np.newaxis]],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]
    )
    return H_transformed


def pohang_2_intrinsics_extrinsics(intrinsics, extrinsics):
    fov_x_left, fov_y_left = get_fov(
        intrinsics["stereo_left"]["focal_length"],
        intrinsics["stereo_left"]["focal_length"],
        intrinsics["stereo_left"]["image_width"],
        intrinsics["stereo_left"]["image_height"],
        type="deg",
    )
    fov_x_right, fov_y_right = get_fov(
        intrinsics["stereo_right"]["focal_length"],
        intrinsics["stereo_right"]["focal_length"],
        intrinsics["stereo_right"]["image_width"],
        intrinsics["stereo_right"]["image_height"],
        type="deg",
    )
    intrinsics = {
        "stereo_left": {
            "fx": intrinsics["stereo_left"]["focal_length"],
            "fy": intrinsics["stereo_left"]["focal_length"],
            "cx": intrinsics["stereo_left"]["cc_x"],
            "cy": intrinsics["stereo_left"]["cc_y"],
            "distortion_coefficients": intrinsics["stereo_left"][
                "distortion_coefficients"
            ],
            "image_width": intrinsics["stereo_left"]["image_width"],
            "image_height": intrinsics["stereo_left"]["image_height"],
            "h_fov": fov_x_left,
            "v_fov": fov_y_left,
        },
        "stereo_right": {
            "fx": intrinsics["stereo_right"]["focal_length"],
            "fy": intrinsics["stereo_right"]["focal_length"],
            "cx": intrinsics["stereo_right"]["cc_x"],
            "cy": intrinsics["stereo_right"]["cc_y"],
            "distortion_coefficients": intrinsics["stereo_right"][
                "distortion_coefficients"
            ],
            "image_width": intrinsics["stereo_right"]["image_width"],
            "image_height": intrinsics["stereo_right"]["image_height"],
            "h_fov": fov_x_right,
            "v_fov": fov_y_right,
        },
    }

    qL = extrinsics["stereo_left"]["quaternion"]
    tL = extrinsics["stereo_left"]["translation"]
    qR = extrinsics["stereo_right"]["quaternion"]
    tR = extrinsics["stereo_right"]["translation"]
    RL = Rotation.from_quat(qL).as_matrix()
    RR = Rotation.from_quat(qR).as_matrix()

    T_POINTS_WORLD_FROM_LEFT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_LEFT[:3, :3] = RL
    T_POINTS_WORLD_FROM_LEFT[:3, 3] = tL

    T_POINTS_WORLD_FROM_RIGHT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_RIGHT[:3, :3] = RR
    T_POINTS_WORLD_FROM_RIGHT[:3, 3] = tR

    T_POINTS_RIGHT_FROM_LEFT = (
        invert_transformation(T_POINTS_WORLD_FROM_RIGHT) @ T_POINTS_WORLD_FROM_LEFT
    )
    # T = TL @ invert_transformation(TR)
    R_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, :3]
    t_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, 3]

    extrinsics = {
        "rotation_matrix": R_POINTS_RIGHT_FROM_LEFT,
        "translation": t_POINTS_RIGHT_FROM_LEFT,
    }
    return intrinsics, extrinsics


import json


def usvinland_extrinsics():
    data = json.load("usvinland.json")
    extrinsics = data["extrinsics"]

    RL = extrinsics["stereo_left"]["rotation_matrix"]
    # tL = extrinsics["stereo_left"]["translation"]
    RR = extrinsics["stereo_right"]["rotation_matrix"]
    # tR = extrinsics["stereo_right"]["translation"]
    PL = extrinsics["stereo_left"]["projection_matrix"]
    PR = extrinsics["stereo_right"]["projection_matrix"]

    T_POINTS_WORLD_FROM_LEFT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_LEFT[:3, :3] = RL
    # T_POINTS_WORLD_FROM_LEFT[:3, 3] = tL

    T_POINTS_WORLD_FROM_RIGHT = np.diag([1.0] * 4)
    T_POINTS_WORLD_FROM_RIGHT[:3, :3] = RR
    # T_POINTS_WORLD_FROM_RIGHT[:3, 3] = tR

    T_POINTS_RIGHT_FROM_LEFT = (
        invert_transformation(T_POINTS_WORLD_FROM_RIGHT) @ T_POINTS_WORLD_FROM_LEFT
    )
    # T = TL @ invert_transformation(TR)
    R_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, :3]
    t_POINTS_RIGHT_FROM_LEFT = T_POINTS_RIGHT_FROM_LEFT[:3, 3]

    extrinsics = {
        "rotation_matrix": R_POINTS_RIGHT_FROM_LEFT,
        "translation": t_POINTS_RIGHT_FROM_LEFT,
    }


def read_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    return data


def find_lowest_y(f, x, start, end):
    while start < end:
        mid = start + (end - start) // 2
        if mid >= f(x):
            end = mid
        else:
            start = mid + 1
    return start


def create_wateredge_mask(points, H, W):
    # Sort points by x-coordinate
    points = points[np.argsort(points[:, 0])]

    # Interpolate between points to define the edge
    f = interp1d(
        points[:, 0], points[:, 1], kind="linear", assume_sorted=True
    )  # , fill_value='extrapolate')

    # Create an empty mask array
    mask = np.zeros((H, W), dtype=np.uint8)

    # # Iterate over each pixel in the mask array
    # for x in range(W):
    #     for y in range(H):
    #         # Determine if the pixel is below the water edge
    #         if y >= f(x):
    #             mask[y:, x] = 1
    #             break
    # Iterate over each pixel in the mask array
    for x in range(W):
        # Find the lowest y such that y >= f(x)
        lowest_y = find_lowest_y(f, x, 0, H)
        # Set mask[y:, x] to 1
        mask[lowest_y:, x] = 1
    return mask


def remove_obstacles_from_watermask(water_mask, obstacles):
    # remove bbox of obstacles from water mask
    obstacles_mask = np.zeros_like(water_mask)
    for obstacle in obstacles:
        x, y, w, h = obstacle
        x, y, w, h = int(x), int(y), int(w), int(h)
        obstacles_mask[y : y + h, x : x + w] = 1
    not_obstacles_mask = 1 - obstacles_mask

    water_mask = np.logical_and(water_mask, not_obstacles_mask)

    return water_mask


def sync_timestamps(ts1, ts2):
    """Synchronizes timestamps

    Args:
        ts1 np.array: (N,)
        ts2 np.array: (M,)

    Returns:
        ts1,ts2 indexes
    """
    if max(ts1.shape[0], ts2.shape[0]) == ts1.shape[0]:
        ts1_idx = np.searchsorted(ts1, ts2, side="left")
        ts1_idx = np.clip(ts1_idx - 1, 0, len(ts1) - 1)
        ts2_idx = np.arange(0, ts2.shape[0])
    else:
        ts2_idx = np.searchsorted(ts2, ts1, side="left")
        ts2_idx = np.clip(ts2_idx - 1, 0, len(ts2) - 1)
        ts1_idx = np.arange(0, ts1.shape[0])

    return ts1_idx, ts2_idx


def find_closest_numbers_idx(num, arr):
    closest_left = None
    closest_right = None

    for i, x in enumerate(arr):
        if x < num:
            if closest_left is None or abs(x - num) < abs(closest_left - num):
                closest_left = x
                closest_left_idx = i
        elif x > num:
            if closest_right is None or abs(x - num) < abs(closest_right - num):
                closest_right = x
                closest_right_idx = i

    return closest_left_idx, closest_right_idx


def normalize_img(img, scale = 1):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= scale
    img = img.astype(np.uint8)
    return img

def homog(vec):
    return np.concatenate([vec, [1]])

def dehomog(vec):
    return vec[:-1] / vec[-1]