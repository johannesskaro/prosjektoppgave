import json
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from pointcloud import PointCloud
from utilities import get_fov


class StereoCam:
    def __init__(self, intrinsics: dict, extrinsics: dict):
        """
        Required format of intrinsics and extrinsics:

        intrinsics = { "stereo_left": { "cx": 0, "cy": 0, "fx": 0, "fy": 0, "distortion_coefficients": [0, 0, 0, 0], "h_fov": 0, "v_fov": 0, "image_width": 0, "image_height": 0 },
                       "stereo_right": { "cx": 0, "cy": 0, "fx": 0, "fy": 0, "distortion_coefficients": [0, 0, 0, 0], "h_fov": 0, "v_fov": 0, "image_width": 0, "image_height": 0 }
                    }
        extrinsics = { "rotation_matrix": [0, 0, 0, 0, 0, 0, 0, 0, 0], "translation": [0, 0, 0] }

        """

        self.intrinsicsL = intrinsics["stereo_left"]
        self.intrinsicsR = intrinsics["stereo_right"]
        self.extrinsics = extrinsics

        cxL, cyL, fxL, fyL, dist_coeffsL, WL, HL = self.parse_intrinsics("left")
        cxR, cyR, fxR, fyR, dist_coeffsR, WR, HR = self.parse_intrinsics("right")

        assert WL == WR and HL == HR, "Image dimensions must be the same"
        W, H = WL, HL

        if "h_fov" not in self.intrinsicsL:
            self.intrinsicsL["h_fov"], self.intrinsicsL["v_fov"] = get_fov(
                fxL, fyL, W, H, type="deg"
            )
            self.intrinsicsR["h_fov"], self.intrinsicsR["v_fov"] = get_fov(
                fxR, fyR, W, H, type="deg"
            )

        KL = self.get_camera_matrix(fxL, fyL, cxL, cyL)
        KR = self.get_camera_matrix(fxR, fyR, cxR, cyR)

        R, t = self.get_extrinsics()
        self.baseline = np.linalg.norm(t)

        self.left_projection_matrix = KL @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.right_projection_matrix = KR @ np.hstack((R, t.reshape(3, 1)))

        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            KL, dist_coeffsL, KR, dist_coeffsR, (W, H), R, t, alpha=0
        )

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            KL, dist_coeffsL, R1, P1, (W, H), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            KR, dist_coeffsR, R2, P2, (W, H), cv2.CV_32FC1
        )

        num_disparities = int(W * 0.08)
        block_size = 4
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=0,
            uniquenessRatio=0,
            speckleWindowSize=200,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM,
        )

    def parse_intrinsics(self, side: str = "left"):
        if side == "left":
            intr = self.intrinsicsL
        if side == "right":
            intr = self.intrinsicsR

        cx, cy = intr["cx"], intr["cy"]
        fx, fy = intr["fx"], intr["fy"]
        d = np.array(intr["distortion_coefficients"])
        W, H = intr["image_width"], intr["image_height"]
        return cx, cy, fx, fy, d, W, H

    def parse_extrinsics(self):
        R = self.extrinsics["rotation_matrix"]
        t = self.extrinsics["translation"]
        return np.array(R), np.array(t)

    def get_basic_camera_parameters(self, side: str = "left"):
        cx, cy, fx, fy, d, W, H = self.parse_intrinsics(side)
        cam_params = {"cx": cx, "cy": cy, "fx": fx, "fy": fy, "b": self.baseline}
        h_fov, v_fov = self.get_fov(side)
        cam_params["h_fov"] = h_fov
        cam_params["v_fov"] = v_fov
        return cam_params

    def get_left_projection_matrix(self):
        return self.left_projection_matrix

    def get_extrinsics(self):
        R, t = self.parse_extrinsics()
        return R, t

    def get_fov(self, side: str = "left"):
        """
        Get field of view in degrees.
        """
        if side == "left":
            intr = self.intrinsicsL
        if side == "right":
            intr = self.intrinsicsR
        h_fov, v_fov = intr["h_fov"], intr["v_fov"]
        return h_fov, v_fov

    def get_disparity(
        self, left_image, right_image, rectify_images=True, stereo_matcher=None
    ):
        """
        Get disparity map from stereo image pair.
        """

        if rectify_images:
            left_image, right_image = self.rectify_images(left_image, right_image)

        if stereo_matcher is None:
            stereo_matcher = self.stereo_matcher

        disparity_SGBM = stereo_matcher.compute(left_image, right_image)
        disparity = disparity_SGBM.astype(np.float32) / 16.0
        # disparity[disparity > disparity.max()*0.7] = 0

        return disparity

    def get_depth_map(
        self, left_image, right_image, rectify_images=True, stereo_matcher=None
    ):
        """
        Retrieves the depth image from an stereo image pair.
        """

        if rectify_images:
            left_image, right_image = self.rectify_images(left_image, right_image)
        if stereo_matcher is None:
            stereo_matcher = self.stereo_matcher

        disparity = self.get_disparity(
            left_image, right_image, rectify_images=False, stereo_matcher=stereo_matcher
        )

        f = self.intrinsicsL["fx"]

        depth_img = self.baseline * f / disparity

        depth_img[depth_img > 100] = 0

        # point_image = cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=False)

        # valid_indeces = np.logical_and(np.all(np.isfinite(point_image), axis=2), point_image[:,:,2] < 150000)

        # depth_img = np.zeros(point_image.shape[:2], dtype=np.float32)

        # depth_img[valid_indeces] = point_image[valid_indeces][:,2]

        return depth_img

    def get_pc(self, left_image, right_image, stereo_matcher=None):
        """
        Retrieves the point cloud from an unrectified stereo image pair.

        """
        left_image, right_image = self.rectify_images(left_image, right_image)

        disparity = self.get_disparity(
            left_image, right_image, rectify_images=False, stereo_matcher=stereo_matcher
        )

        point_image = cv2.reprojectImageTo3D(
            disparity, self.Q, handleMissingValues=False
        )

        valid_indeces = np.logical_and(
            np.all(np.isfinite(point_image), axis=2), point_image[:, :, 2] < 100000
        )
        pc_xyz = point_image[valid_indeces]
        pc_bgr = left_image[valid_indeces] / 255
        pc_rgb = np.fliplr(pc_bgr)

        return pc_xyz, pc_rgb

    def rectify_images(self, imgL, imgR):

        imgL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LINEAR)
        imgR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LINEAR)
        # imgL = cv2.remap(imgL, self.map1x, self.map1y, cv2.INTER_LANCZOS4)
        # imgR = cv2.remap(imgR, self.map2x, self.map2y, cv2.INTER_LANCZOS4)

        return imgL, imgR

    def get_camera_matrix(self, fx, fy, cx, cy):
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K