# Load data for the experiment from the summer of 2023 with multiple ZED cameras. 

import numpy as np
import pymap3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

import pyzed.sl as sl

import sys
sys.path.insert(0, "/home/nicholas/GitHub/phd-stereo/analysis/ma2_rosbag")
from extrinsics import H_POINTS_LEFT_CAM_FROM_FLOOR, H_POINTS_LEFT_ZED_FROM_LEFT_CAM, H_POINTS_PIREN_ENU_FROM_PIREN, H_POINTS_VESSEL_FROM_FLOOR, PIREN_ALT, PIREN_LAT, PIREN_LON, invert_transformation
sys.path.insert(0, "/home/nicholas/GitHub/phd-stereo/python_tools")


GNSS_TOPIC = "/senti_parser/SentiPose"
LIDAR_TOPIC = "/lidar_aft/points"

def transform_enu_from_left_zed(gnss_timestamp, pc_xyz_zed, all_ma2_pose_ned):
    ma2_pose_ned = get_row_from_timestamp(gnss_timestamp, all_ma2_pose_ned)
    H_points_enu_from_zed = get_transform_enu_from_left_zed(ma2_pose_ned)
    pc_xyz_enu = homogeneous_multiplication(H_points_enu_from_zed, pc_xyz_zed)
    return pc_xyz_enu

def get_transform_enu_from_left_zed(ma2_pose_ned):
    H_points_piren_from_vessel = get_transform_piren_ned_from_vessel(ma2_pose_ned)
    H_POINTS_FLOOR_FROM_LEFT_CAM = invert_transformation(H_POINTS_LEFT_CAM_FROM_FLOOR)
    H_POINTS_LEFT_CAM_FROM_LEFT_ZED = invert_transformation(H_POINTS_LEFT_ZED_FROM_LEFT_CAM)
    H_points_enu_from_zed = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel @ H_POINTS_VESSEL_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_LEFT_ZED
    return H_points_enu_from_zed

def get_transform_piren_ned_from_vessel(ma2_pose_ned):
    ma2_trans = ma2_pose_ned[1:4]
    ma2_rot_quat = ma2_pose_ned[4:]
    ma2_rot_mat = R.from_quat(ma2_rot_quat).as_matrix()

    H_points_piren_from_vessel = np.block([
        [ma2_rot_mat, ma2_trans[:,np.newaxis]], 
        [np.zeros((1,3)), np.ones((1,1))]
    ])
    return H_points_piren_from_vessel

def homogeneous_multiplication(H_np1xnp1, pts_mxn):
        m, n = pts_mxn.shape
        np1 = H_np1xnp1.shape[0]
        assert H_np1xnp1.shape[0] == H_np1xnp1.shape[1]
        assert n+1 == np1

        pts_t = H_np1xnp1.dot(np.r_[pts_mxn.T, np.ones((1, m))])[0:n, :].T
        return pts_t

def get_row_from_timestamp(timestamp, table):
    idx = get_row_idx_from_timestamp(timestamp, table)
    return table[idx]

def get_row_idx_from_timestamp(timestamp, table):
    idx = np.searchsorted(table[:,0], timestamp, "left")
    if idx == 0: 
        print(f"Timestamp before start of table. ")
        return 0
    if idx == table.shape[0]: 
        print(f"Timestamp after end of table. ")
        return -1
    if abs(table[idx-1][0] - timestamp) < abs(table[idx][0] - timestamp):
        return idx-1
    else:
        return idx
        
def get_all_ma2_pose_ned(rosbag_path):
    t_pos_ori = []
    with Reader(f"{rosbag_path}") as reader:
        connections = [c for c in reader.connections if c.topic == GNSS_TOPIC]
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            timestamp_msg = msg.header.stamp.sec * (10**9) + msg.header.stamp.nanosec

            pos_ros = msg.pose.position
            pos = np.array([pos_ros.x, pos_ros.y, pos_ros.z])
            ori_ros = msg.pose.orientation
            ori_quat = np.array([ori_ros.x, ori_ros.y, ori_ros.z, ori_ros.w])

            t_pos_ori.append([timestamp_msg, *pos, *ori_quat])
    # pos is here relative to piren, which is NED
    return np.array(t_pos_ori)

def gen_lidar_pc(rosbag_path):
    with Reader(f"{rosbag_path}") as reader:
        connections = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        assert len(connections) == 1
        
        for connection, timestamp, rawdata in reader.messages(connections):
            msg = deserialize_cdr(rawdata, connection.msgtype)
            timestamp_msg = msg.header.stamp.sec * (10**9) + msg.header.stamp.nanosec
            
            lidar_xyz = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
            lidar_rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (lidar_xyz.shape[0], 1))

            yield timestamp_msg, lidar_xyz, lidar_rgb

def get_kb2_gnss_enu(kb2_path):
    df = pd.read_table(kb2_path, sep='\s+', header=24, parse_dates={'Timestamp': [0, 1]})

    df_vals = df["Timestamp"].astype(int).to_numpy(), df["longitude(deg)"].to_numpy(), df["latitude(deg)"].to_numpy(), df["height(m)"].to_numpy()
    ps_geo =[]
    for ts, long, lat, height in np.array(df_vals).T:
        # The KB2 box requires handling of leap seconds. 
        # See: https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python
        ts_with_leap = ts + (-37+19) * 10**9
        ps_geo.append([ts_with_leap, long, lat, height])
    pts = np.array(ps_geo)
    ps_enu = lon_lat_to_xy_enu(pts)
    return ps_enu

def get_sd_gnss_enu(sd_path):
    df = pd.read_csv(sd_path)
    df_vals = (df["Unix Time"] * (10**6) + df["Microseconds"]).to_numpy(), df["Longitude (degrees)"].to_numpy(), df["Latitude (degrees)"].to_numpy(), df["Height (m)"].to_numpy(), df["Velocity East (m/s)"].to_numpy(), df["Velocity North (m/s)"].to_numpy()
    ps_geo =[]
    for ts, long, lat, height, ulong, ulat in np.array(df_vals).T:
        # ts_with_leap = ts + (-37+19) * 10**9
        ps_geo.append([ts*(10**3), long, lat, height]) # SD gives milliseconds
    ps_enu = lon_lat_to_xy_enu(np.array(ps_geo))
    return ps_enu

def get_nt_gnss_enu(nt_path):
    df = pd.read_table(nt_path, sep='\s+', header=24, parse_dates={'Timestamp': [0, 1]})

    df_vals = df["Timestamp"].astype(int).to_numpy(), df["longitude(deg)"].to_numpy(), df["latitude(deg)"].to_numpy(), df["height(m)"].to_numpy()
    ps_geo =[]
    for ts, long, lat, height in np.array(df_vals).T:
        # The NTBox requires handling of leap seconds. 
        # See: https://stackoverflow.com/questions/33415475/how-to-get-current-date-and-time-from-gps-unsegment-time-in-python
        ts_with_leap = ts + (-37+19) * 10**9
        ps_geo.append([ts_with_leap, long, lat, height])
    pts = np.array(ps_geo)
    ps_enu = lon_lat_to_xy_enu(pts)
    return ps_enu

def lon_lat_to_xy_enu(t_lon_lat_alt):
    ps = []
    for p_geo in t_lon_lat_alt:
        t, lon, lat, alt = p_geo
        x, y, h = pymap3d.geodetic2enu(lat, lon, alt, PIREN_LAT, PIREN_LON, PIREN_ALT)
        ps.append([t, x, y, h]) # t, ENU
    return np.array(ps)
