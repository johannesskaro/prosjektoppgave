import open3d as o3d
import numpy as np
from utilities import calculate_3d_points, calculate_3d_point


class PointCloud:

    def __init__(self, xyz = None, rgb = None):
        self.pc = o3d.geometry.PointCloud()

        if xyz is not None and rgb is not None:
            self.set_xyz(xyz)
            self.set_rgb(rgb)
        else:
            self.set_xyz(np.zeros((1,3)))
            self.set_rgb(np.zeros((1,3)))

    def set_xyz(self, xyz: np.array) -> None:
        self.pc.points = o3d.utility.Vector3dVector(xyz)

    def set_rgb(self, rgb: np.array) -> None:
        self.pc.colors = o3d.utility.Vector3dVector(rgb)

    def set_xyzrgb(self, xyz: np.array, rgb: np.array) -> None:
        self.set_xyz(xyz)
        self.set_rgb(rgb)

    def set_pc(self, pc: o3d.geometry.PointCloud) -> None:
        self.pc = pc

    def create_from_img_and_depth(self, img: np.array, depth_img: np.array, cam_params: dict) -> o3d.geometry.PointCloud:
        """
        Set the point cloud from an image and a depth map.
        """
        if img.shape[:2] != depth_img.shape:
            raise ValueError("Image and depth map must have the same dimensions")

        (H,W) = depth_img.shape
        img_o3d = o3d.geometry.Image(img)
        depth_o3d = o3d.geometry.Image(depth_img)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(W,H, 
                                                       cam_params["fx"], 
                                                       cam_params["fy"], 
                                                       cam_params["cx"], 
                                                       cam_params["cy"])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_o3d, depth_o3d, depth_scale=1.0, depth_trunc=200.0, convert_rgb_to_intensity=False)
        self.pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, project_valid_depth_only=False)
        
        return self

    def get_o3d_pc(self) -> o3d.geometry.PointCloud:
        return self.pc