from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import pyzed.sl as sl

import sys
#sys.path.insert(0, r"C:\Users\johro\Documents\BB-Perception\2023-summer-experiment\python_tools")
sys.path.insert(0, "/home/johannes/Documents/blueboats/prosjektoppgave/python_tools")
from stereo_svo import SVOCamera

#Scen1 - Into tunnel
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-30-51_28170706_HD1080_FPS15.svo"
#ROSBAG_NAME = "scen1"
#START_TIMESTAMP = 1689067892194593719 + 120000000000
#ma2_clap_timestamps = np.array([1689068801634572145, 1689068803035078922, 1689068804635190937, 1689068806436892969, 1689068809235474632]) 
#svo_clap_timestamps = np.array([1689068801796052729, 1689068803135787729, 1689068804743766729, 1689068806686255729, 1689068809298756729]) 


#Scen2_2 - Crossing
SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_11-52-01_28170706_HD1080_FPS15.svo"
ROSBAG_NAME = "scen2_2"
START_TIMESTAMP = 1689069183452515635 + 25000000000
svo_clap_timestamps = np.array([1689069149608491571, 1689069151551222571,1689069153962745571,1689069155369492571, 1689069156977375571, 1689069158785868571]) 
ma2_clap_timestamps = np.array([1689069149450584626, 1689069151450466830,1689069153851288364,1689069155051366874, 1689069156851162758, 1689069158651091124])


#Scen4_2 - Docking w. boats
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_28170706_HD1080_FPS15.svo" #right zed 
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-20-43_5256916_HD1080_FPS15.svo" #left zed
#ROSBAG_NAME = "scen4_2"
#START_TIMESTAMP = 1689070899731613030
#START_TIMESTAMP = 1689070888907352002# Starting to see kayak
#START_TIMESTAMP = 1689070910831613030 #Docking
#ma2_clap_timestamps = np.array([1689070864130009197, 1689070865931143443, 1689070867729428949, 1689070870332243623, 1689070872330384680])
#svo_clap_timestamps = np.array([1689070864415441257, 1689070866090016257, 1689070867898886257, 1689070870444290257, 1689070872386914257]) 


#Scen5 - Docking with tube
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-49-30_28170706_HD1080_FPS15.svo" #port side zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-49-30_5256916_HD1080_FPS15.svo" #starboard side zed
#ROSBAG_NAME = "scen5"
#START_TIMESTAMP = 1689072610543382582
#START_TIMESTAMP = 1689072601809970113
#START_TIMESTAMP = 1689072611943528062
#START_TIMESTAMP = 1689072633543528062
#ma2_clap_timestamps = np.array([(1689072578409069874 + 1689072578610974426) / 2,1689072580008516107,1689072581409199315,1689072582609241408,(1689072584209923187 + 1689072584409875973) / 2,1689072585209757240,])
#svo_clap_timestamps = np.array([1689072578584773729,1689072580192777729,1689072581532534729,(1689072582805224729 + 1689072582872153729) / 2,1689072584412922729,(1689072585350708729 + 1689072585417710729) / 2,])


#Scen6 - Docking with tube further away
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-55-58_28170706_HD1080_FPS15.svo" #port side zed
#SVO_FILE_PATH = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\ZED camera svo files\2023-07-11_12-55-58_5256916_HD1080_FPS15.svo" # left zed
#ROSBAG_NAME = "scen6"
#START_TIMESTAMP = 1689073008428931880# Starting to see tube
#START_TIMESTAMP = 1689073018428931880 # tube almost passed
#START_TIMESTAMP = 1689073021428931880 # tube passed
#ma2_clap_timestamps = np.array([1689072978427718986, 1689072980427686560, 1689072982230896164, 1689072984228220707])
#svo_clap_timestamps = np.array([1689072978666263269, 1689072980675916269, 1689072982484494269, 1689072984360142269])


diffs_s = (ma2_clap_timestamps - svo_clap_timestamps) / (10 ** 9)
GNSS_MINUS_ZED_TIME_NS = np.mean(ma2_clap_timestamps - svo_clap_timestamps)
ROSBAG_FOLDER = r"C:\Users\johro\Documents\2023-07-11_Multi_ZED_Summer\bags"
ROSBAG_PATH = f"{ROSBAG_FOLDER}/{ROSBAG_NAME}"


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
ROT_FLOOR_TO_LEFT_CAM = np.array([[-3.20510335e-09, -3.20510335e-09, -1.00000000e+00],
       [-1.00000000e+00, -5.55111512e-17,  3.20510335e-09],
       [-5.55111512e-17,  1.00000000e+00, -3.20510335e-09]])
TRANS_FLOOR_TO_LEFT_CAM = np.array([-4.1358,  1.0967, -0.702 ])

H_POINTS_LEFT_CAM_FROM_FLOOR = np.block([
    [ROT_FLOOR_TO_LEFT_CAM.T, -ROT_FLOOR_TO_LEFT_CAM.T.dot(TRANS_FLOOR_TO_LEFT_CAM)[:,np.newaxis]], 
    [np.zeros((1,3)), np.ones((1,1))]
])


H_POINTS_LEFT_ZED_FROM_LIDAR = H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_LEFT_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR
#H = H_POINTS_LEFT_ZED_FROM_LIDAR  #Use left zed
H = H_POINTS_RIGHT_FROM_LEFT_ZED @ H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR #Use right zed
H_INV_TO_RIGHT_ZED = np.linalg.inv(H_POINTS_LEFT_ZED_FROM_LEFT_CAM @ H_POINTS_CAM_FROM_FLOOR @ H_POINTS_FLOOR_FROM_LIDAR)



LIDAR_TOPIC = "/lidar_aft/points"

stereo_cam = SVOCamera(SVO_FILE_PATH)
stereo_cam.set_svo_position_timestamp(START_TIMESTAMP)

K, D = stereo_cam.get_left_parameters()
_, _, R, T = stereo_cam.get_right_parameters()
focal_length = K[0,0]
baseline = np.linalg.norm(T)

def gen_svo_images():
    num_frames = 275
    curr_frame = 0
    while stereo_cam.grab() == sl.ERROR_CODE.SUCCESS and curr_frame < num_frames:
        image = stereo_cam.get_left_image(should_rectify=True)
        timestamp = stereo_cam.get_timestamp()
        disparity_img = stereo_cam.get_neural_disp()
        #disparity_img = stereo_cam.get_disparity()
        print(f"Curr frame: {curr_frame}")
        curr_frame += 1
        #depth_img = baseline * focal_length / disparity_img
        depth_img = stereo_cam.get_depth_image()

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
                
                xyz_c_forward = xyz_c[xyz_c[:,2] > 0]
                image_points_forward = image_points[xyz_c[:,2] > 0]
                intensity_clipped_forward = intensity_clipped[xyz_c[:,2] > 0]

                yield timestamp, image_points_forward, intensity_clipped_forward, xyz_c_forward

def transform_from_image_plane_to_3d(xyz_c):

    xyz_image_homogeneous = np.r_[xyz_c.T, np.ones((1, xyz_c.shape[0]))]  # shape (4, N)
    xyz_right_zed_homogeneous = H_POINTS_RIGHT_FROM_LEFT_ZED @ H_INV_TO_RIGHT_ZED.dot(xyz_image_homogeneous)
    xyz_right_zed = (xyz_right_zed_homogeneous[0:3, :] / xyz_right_zed_homogeneous[3, :]).T  # Shape: (N, 3)

    return xyz_right_zed
