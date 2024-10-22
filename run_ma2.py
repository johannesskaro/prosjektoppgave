import open3d as o3d
import numpy as np
from scipy.io import loadmat
import time

# Load the .mat file containing LiDAR points
mat_file = loadmat('data/lidar_pts_scen4_2.mat')

# Assuming 'lidar_aft_pts' contains a sequence of LiDAR point cloud data
# Each element in 'lidar_aft_pts' could be [timestamp, points]
lidar_data = mat_file['lidar_aft_pts']

# Create a visualizer object for real-time updates
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()

# Iterate over the sequence of point clouds
for i, data in enumerate(lidar_data):
    timestamp, points = data[0], data[1]  # Extract timestamp and point cloud

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Convert points to a NumPy array if they aren't already
    points_array = np.array(points)
    
    # Assign the points to the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(points_array)

    # Clear the previous point cloud and add the new one
    vis.clear_geometries()
    vis.add_geometry(pcd)
    
    # Update the visualizer window
    vis.poll_events()
    vis.update_renderer()
    
    # Optional delay to control playback speed
    time.sleep(0.5)

# Close the visualizer window
vis.destroy_window()