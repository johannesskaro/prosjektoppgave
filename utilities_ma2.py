from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import pyzed.sl as sl

import sys
sys.path.insert(0, r"C:\Users\johro\Documents\BB-Perception\2023-summer-experiment\python_tools")
from stereo_svo import SVOCamera

#SVO_FILE_PATH = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2\svo-files\scen6\2023-07-11_12-55-58_5256916_HD1080_FPS15.svo"
#SVO_FILE_PATH = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2\svo-files\scen5\2023-07-11_12-49-30_5256916_HD1080_FPS15.svo" #starboard side zed
SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-49-30_28170706_HD1080_FPS15.svo" #port side zed
ROSBAG_FOLDER = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2"
ROSBAG_NAME = "scen5" #scen6 is docking 2
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"

GNSS_MINUS_ZED_TIME_NS = -201327414.75 # Docking 2
# GNSS_MINUS_ZED_TIME_NS = -136682008.0 # Buster maneuver
# GNSS_MINUS_ZED_TIME_NS = -156467678.6 # Undocking 2

START_TIMESTAMP = 1689072610543382582
END_TIMESTAMP = 1689072610543382582 + 10**9

ROT_FLOOR_TO_LIDAR = np.array([[-8.27535228e-01,  5.61392452e-01,  4.89505779e-03],
       [ 5.61413072e-01,  8.27516685e-01,  5.61236993e-03],
       [-8.99999879e-04,  7.39258326e-03, -9.99972269e-01]])
TRANS_FLOOR_TO_LIDAR = np.array([-4.1091, -1.1602, -1.015 ])
H_POINTS_FLOOR_FROM_LIDAR = np.block([
    [ROT_FLOOR_TO_LIDAR, TRANS_FLOOR_TO_LIDAR[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

# Using left MA2 cam pose
ROT_FLOOR_TO_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
TRANS_FLOOR_TO_CAM = np.array([-4.1358,  1.0967, -0.702 ])

H_POINTS_CAM_FROM_FLOOR = np.block([
    [ROT_FLOOR_TO_CAM.T, -ROT_FLOOR_TO_CAM.T.dot(TRANS_FLOOR_TO_CAM)[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])

ROT_EXTRINSIC = np.array([[ 0.99940448, -0.0263474 ,  0.02228227],
                          [ 0.02664859,  0.99955599, -0.01332975],
                          [-0.02192117,  0.01391561,  0.99966285]])
TRANS_EXTRINSIC = np.array([[-1.95],
                            [-0.02335371],
                            [ 0.2]])
ROT_EXTRINSIC = R.from_euler("y", [-2], degrees=True).as_matrix()[0] @ ROT_EXTRINSIC
H_POINTS_RIGHT_FROM_LEFT_ZED = np.block([
    [ROT_EXTRINSIC, TRANS_EXTRINSIC], 
    [np.zeros((1,3)), np.ones((1,1))]
])

H_POINTS_LEFT_ZED_FROM_LEFT_CAM = np.array([
    [ 0.99987632,  0.00293081,  0.01545154, -0.04023359],
    [-0.00137582,  0.99501634, -0.09970255,  0.353144  ],
    [-0.01566675,  0.09966896,  0.99489731, -0.04098881],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

H = H_POINTS_RIGHT_FROM_LEFT_ZED @ H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR

LIDAR_TOPIC = "/lidar_aft/points"

stereo_cam = SVOCamera(SVO_FILE_PATH)
stereo_cam.set_svo_position_timestamp(START_TIMESTAMP - GNSS_MINUS_ZED_TIME_NS)

K, D = stereo_cam.get_left_parameters()
_, _, R, T = stereo_cam.get_right_parameters()
focal_length = K[0,0]
baseline = np.linalg.norm(T)

def gen_svo_images():
    num_frames = 300
    curr_frame = 0
    while stereo_cam.grab() == sl.ERROR_CODE.SUCCESS and curr_frame < num_frames:
        image = stereo_cam.get_left_image(should_rectify=True)
        timestamp = stereo_cam.get_timestamp()
        disparity_img = stereo_cam.get_neural_disp()
        #disparity_img = stereo_cam.get_disparity()
        curr_frame += 1
        depth_img = baseline * focal_length / disparity_img

        yield timestamp + GNSS_MINUS_ZED_TIME_NS, image, disparity_img, depth_img


def gen_ma2_lidar_points():
    with Reader(ROSBAG_PATH) as reader:
        connections = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
        assert len(connections) == 1
        
        for connection, timestamp, rawdata in reader.messages(connections):
            if timestamp > START_TIMESTAMP:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                xyz = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
                intensity = msg.data.reshape(-1, msg.point_step)[:,16:20].view(dtype=np.float32)
                rgb = np.tile(np.array([255, 0, 0], dtype=np.uint8)/255, (xyz.shape[0], 1))

                intensity_clipped = np.clip(intensity, 0, 100)

                #plt.clf()
                #plt.hist(intensity_clipped, bins=100)
                #plt.pause(0.1)
                
                xyz_c = H.dot(np.r_[xyz.T, np.ones((1, xyz.shape[0]))])[0:3, :].T
              
                rvec = np.zeros((1,3), dtype=np.float32)
                tvec = np.zeros((1,3), dtype=np.float32)
                distCoeff = np.zeros((1,5), dtype=np.float32)
                image_points, _ = cv2.projectPoints(xyz_c, rvec, tvec, K, distCoeff)

                image_points_forward = image_points[xyz_c[:,2] > 0]
                intensity_clipped_forward = intensity_clipped[xyz_c[:,2] > 0]

                yield timestamp, image_points_forward, intensity_clipped_forward

