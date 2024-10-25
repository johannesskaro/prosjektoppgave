from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np

import pyzed.sl as sl

import sys
sys.path.insert(0, r"C:\Users\johro\Documents\BB-Perception\2023-summer-experiment\python_tools")
from stereo_svo import SVOCamera

#SVO_FILE_PATH = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2\svo-files\scen6\2023-07-11_12-55-58_5256916_HD1080_FPS15.svo"
SVO_FILE_PATH = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2\svo-files\scen5\2023-07-11_12-49-30_5256916_HD1080_FPS15.svo"
ROSBAG_FOLDER = r"C:\Users\johro\Documents\BB-Perception\Datasets\ma2"
ROSBAG_NAME = "scen5" #scen6 is docking 2
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"

# ma2_clap_timestamps = np.array([1689068801634572145, 1689068803035078922, 1689068804635190937, 1689068806436892969, 1689068809235474632]) # Buster maneuver, "timestamp" from rosbags package
# ma2_clap_timestamps = np.array([1689068838573357168, 1689068839973421904, 1689068841573496152, 1689068843373579480, 1689068846173708880]) # Buster maneuver, gnss time, wrong
ma2_clap_timestamps = np.array([1689072978427718986, 1689072980427686560, 1689072982230896164, 1689072984228220707])  # Docking 2, from "timestamp"
# ma2_clap_timestamps = np.array([1689070864130009197, 1689070865931143443, 1689070867729428949, 1689070870332243623, 1689070872330384680])  # Undocking 2, right

# svo_clap_timestamps = np.array([1689068801796052729, 1689068803135787729, 1689068804743766729, 1689068806686255729, 1689068809298756729])  # Buster maneuver
svo_clap_timestamps = np.array([1689072978599279269, 1689072980675916269, 1689072982484494269, 1689072984360142269])  # Docking 2
# svo_clap_timestamps = np.array([1689070864415441257, 1689070866090016257, 1689070867898886257, 1689070870444290257, 1689070872386914257])  # Undocking 2, right

diffs_s = (ma2_clap_timestamps - svo_clap_timestamps) / (10 ** 9)
GNSS_MINUS_ZED_TIME_NS = np.mean(ma2_clap_timestamps - svo_clap_timestamps)

START_TIMESTAMP = 1689072610543382582
END_TIMESTAMP = 1689072610543382582 + 10**9

ROT_FLOOR_TO_LIDAR = np.array([[-8.27535228e-01,  5.61392452e-01,  4.89505779e-03],
       [ 5.61413072e-01,  8.27516685e-01,  5.61236993e-03],
       [-8.99999879e-04,  7.39258326e-03, -9.99972269e-01]])
TRANS_FLOOR_TO_LIDAR = np.array([-4.1091, -1.1602, -1.015 ])

# Guesstimating the camera(?)
ROT_FLOOR_TO_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
#TRANS_FLOOR_TO_CAM = np.array([-4.05,  0.9, -1.1 ])
TRANS_FLOOR_TO_CAM = np.array([-4.1358,  1.0967, -0.702 ])

LIDAR_TOPIC = "/lidar_aft/points"
K = np.loadtxt(f"scen6_calibration/K_matrix.txt")
T = np.loadtxt(f"scen6_calibration/T_matrix.txt")
f = K[0,0]
baseline = np.linalg.norm(T)

def gen_svo_images():
    stereo_cam = SVOCamera(SVO_FILE_PATH)
    stereo_cam.set_svo_position_timestamp(START_TIMESTAMP - GNSS_MINUS_ZED_TIME_NS)
    num_frames = 200
    curr_frame = 0

    while stereo_cam.grab() == sl.ERROR_CODE.SUCCESS and curr_frame < num_frames:
        image = stereo_cam.get_left_image(should_rectify=True)
        timestamp = stereo_cam.get_timestamp()
        disparity_img = stereo_cam.get_neural_disp()
        #disparity_img = stereo_cam.get_disparity()
        curr_frame += 1

        yield timestamp + GNSS_MINUS_ZED_TIME_NS, image, disparity_img


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

                # Transform points from lidar to floor
                transform = np.block([
                    [ROT_FLOOR_TO_LIDAR, TRANS_FLOOR_TO_LIDAR[:,np.newaxis]], 
                    [np.zeros((1,3)), np.ones((1,1))]
                ])
                xyz_f = transform.dot(np.r_[xyz.T, np.ones((1, xyz.shape[0]))])[0:3, :].T

                # Transform points from floor to cam
                transform = np.block([
                    [ROT_FLOOR_TO_CAM.T, -ROT_FLOOR_TO_CAM.T.dot(TRANS_FLOOR_TO_CAM)[:,np.newaxis]], 
                    [np.zeros((1,3)), np.ones((1,1))]
                ])
                xyz_c = transform.dot(np.r_[xyz_f.T, np.ones((1, xyz_f.shape[0]))])[0:3, :].T

                rvec = np.zeros((1,3), dtype=np.float32)
                tvec = np.zeros((1,3), dtype=np.float32)
                distCoeff = np.zeros((1,5), dtype=np.float32)
                image_points, _ = cv2.projectPoints(xyz_c, rvec, tvec, K, distCoeff)

                image_points_forward = image_points[xyz_c[:,2] > 0]
                intensity_clipped_forward = intensity_clipped[xyz_c[:,2] > 0]

                yield timestamp, image_points_forward, intensity_clipped_forward

