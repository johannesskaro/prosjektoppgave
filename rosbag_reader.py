# Mainly used for reading rosbag2 files and saving desired topics to .mat file or python object
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
import struct
from scipy.io import savemat
import numpy as np
import cv2
from extrinsics import H_POINTS_PIREN_ENU_FROM_PIREN 
from scipy.spatial.transform import Rotation

class RosbagData:
    lidar_aft_pts = [] #  timestamp [s] (float) , xyz points [m] (list[float,float,float])
    raw_imgs = [] # timestamp [s] (float) - np.ndarray pairs
    gnss_pose = [] # timestamp [s] (float) - position [x,y,z] (np.ndarray) - orientation [x,y,z,w] (np.ndarray)
    K = np.array([]) # camera intrinsics matrix

typestore = get_typestore(Stores.ROS2_FOXY)

# rosbag config
ROSBAG_FOLDER = '/Users/johannesskaro/Documents/KYB 5.år/Datasets/ma2'
#r'C:\Users\aflaptop\Documents\wesenberg-semester_project\2023-07-11_Multi_ZED_Summer\MA2'
ROSBAG_NAME = 'scen5'
START_TIMESTAMP = 1689070888.907352002

# topic configs
IMG_RAW_TOPIC = '/rgb/ap_a/image_raw'
LIDAR_TOPIC = '/lidar_aft/points'
CAM_INFO_TOPIC = '/rgb/ap_a/camera_info'
GNSS_TOPIC = '/senti_parser/SentiPose'
TOPICS =  [IMG_RAW_TOPIC, CAM_INFO_TOPIC, GNSS_TOPIC, LIDAR_TOPIC] # Note: its a bit slower when reading lidar data, so for testing it might be easier to comment out lidar topic


def read_rosbag(bag_path: Path) -> RosbagData:   
    print("reading rosbag: ", bag_path) 
    rosbag_data = RosbagData()

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        reader_connections = [x for x in reader.connections if x.topic in TOPICS]

        for reader_connection, timestamp, rawdata in reader.messages(connections=reader_connections):
            # print("topic: ", reader_connection.topic)

            msg = reader.deserialize(rawdata, reader_connection.msgtype)
            msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            #print("timediff: ", timestamp*1e-9-msg_timestamp)

            if reader_connection.topic == LIDAR_TOPIC: 
                lidar_pts = read_lidar_pts(msg)
                rosbag_data.lidar_aft_pts.append( [msg_timestamp, lidar_pts ] )

            elif reader_connection.topic == IMG_RAW_TOPIC:
                rosbag_data.raw_imgs.append( [msg_timestamp, np.resize(msg.data, (msg.height, msg.width ))])

            elif reader_connection.topic == CAM_INFO_TOPIC:
                rosbag_data.K = msg.k.reshape([3, 3]) #read in camera intrinsics params
            
            elif reader_connection.topic == GNSS_TOPIC:
                pos, ori = get_ma2_pose_ned(msg)
                rosbag_data.gnss_pose.append([msg_timestamp, pos, ori])
        print("finished reading rosbag")
        return rosbag_data
                    
def read_lidar_pts(msg) -> tuple[   list[float,float,float],    list[float]  ]:
    """
    msg: a PointCloud2 msg containing lidar pts (sensor_msgs__msg__PointCloud2)
    returns a Nx3 list of xyz points in the point cloud
    """
    lidar_pts = []
    
    # for row in range(msg.height):
    #     for col in range(msg.width):
    #         index = (row*msg.row_step) + (col*msg.point_step)
    #         (x,y,z) = struct.unpack_from('fff', msg.data, offset=index)
    #         lidar_pts.append([x,y,z])
    # Faster way to extract data
    lidar_pts = msg.data.reshape(-1, msg.point_step)[:,:12].view(dtype=np.float32)
    lidar_pts = lidar_pts[:,:3]

    return lidar_pts

def get_ma2_pose_ned(msg) -> tuple[   np.ndarray,     np.ndarray  ]:
    # takes in a rosbag msg and converts it to a tuple containing ground truth position (x,y,z in NED frame)
    # and orientation (x,y,z,w quaternion)

    pos = msg.pose.position
    ma2_pos_ned = np.array([ pos.x, pos.y, pos.z]) # position relative to Piren in NED, homogenous
    ori = msg.pose.orientation
    ma2_rot_quat_ned = np.array([ori.x, ori.y, ori.z, ori.w])

    return (ma2_pos_ned, ma2_rot_quat_ned)

def get_transform_piren_ned_from_vessel(ma2_pos_ned, ma2_rot_quat_ned) -> np.ndarray:
    rot_mat = Rotation.from_quat(ma2_rot_quat_ned).as_matrix()

    H_points_piren_from_vessel = np.block([
        [rot_mat, ma2_pos_ned[:,np.newaxis]],
        [np.zeros((1,3)), np.ones((1,1))]
    ])

    return H_points_piren_from_vessel

def get_ma2_pose_enu(ma2_pos_ned, ma2_rot_quat_ned) -> tuple[   np.ndarray,     np.ndarray  ]:
    H_points_piren_from_vessel = get_transform_piren_ned_from_vessel(ma2_pos_ned, ma2_rot_quat_ned)

    H_points_enu_from_vessel = H_POINTS_PIREN_ENU_FROM_PIREN @ H_points_piren_from_vessel

    R_enu = H_points_enu_from_vessel[0:3,0:3]
    rot_quat = Rotation.from_matrix(R_enu).as_quat()
    
    pos_ENU = H_points_enu_from_vessel[0:3,3]

    return (pos_ENU, rot_quat)



def visualize_dataset(imgs: list[str, np.ndarray], inspect_mode=False):
    # inspect mode means that you can visually inspect each frame by clicking back and fourth with the keyboard.
    # This is useful when wanting to visually inspect frames and corresponding timestamps
    num_imgs = imgs.__len__()
    i_frame = 0 #initialize iterator

    # CAMERA
    while i_frame < num_imgs:
        msg_timestamp,image = imgs[i_frame]
        image = cv2.demosaicing(image, cv2.COLOR_BAYER_BG2BGR)

        cv2.imshow(ROSBAG_NAME, image)

        if inspect_mode:
            print("timestamp: ", msg_timestamp)
            key = cv2.waitKey(0)

            if key == 27: return # ESC key
            elif (key == 0) or (key==13): i_frame+=1 # right arrow or enter
            #elif (key == 1) and (i_frame>0): i_frame-=1 # left arrow
            else: print("key pressed: ", key)


        else:
            cv2.waitKey(100)
            i_frame+=1
    



if __name__ == "__main__":
    bag_path = Path(ROSBAG_FOLDER+"/"+ROSBAG_NAME)

    rosbag_data = read_rosbag(bag_path)

    savemat('data/lidar_pts_scen4_2.mat', {'lidar_aft_pts': rosbag_data.lidar_aft_pts})
    print("starting visualization")
    visualize_dataset(rosbag_data.raw_imgs, inspect_mode=False)

    print("finished")

"""
All available topics:
'/initialpose', 
'/navigation/pose', 
'/lidar_fore/points_filtered', 
'/radar/points_filtered', 
'/io_machine/io_driver/do', 
'/lidar_fore/detections', 
'/lidar_fore/detector/points_statistics', 
'/navigation/twist_body', 
'/senti_parser/SentiInputCapture', 
'/goal_pose', 
'/crossing_introspection/smach/container_structure', 
'/senti_parser/SentiGNSSRelPos', 
'/radar/points', 
'/senti_parser/SentiGNSSRfiStatus', 
'/io_machine/io_driver/di', 
'/sitaw/estimates', 
'/sitaw/tracks_as_markers', 
'/senti_parser/SentiNavigationFull', 
'/supervisory/heartbeat', 
'/senti_parser/SentiGNSSNavPvt', 
'/sitaw/tracks', 
'/navigation/twist', 
'/supervisory/enable_autonomy', 
'/rgb/ap_a/image_raw/compressed', 
'/rgb/ap_a/camera_info', 
'/crossing_introspection/smach/container_status', 
'/parameter_events', 
'/supervisory/supervisory_state', 
'/crossing_behavior/quay_vis', 
'/reference_generator/dp_reference', 
'/autonomy_idle_behavior/heartbeat', 
'/targets_as_triangles_markers_topic', 
'/radar/detections', 
'/ravnkloa_lidar_base_map', 
'/events/write_split', 
'/reference_generator/control_selector_mode', 
'/tf_static', 
'/reference_generator/pose_move', 
'/InteractiveVertices/feedback', 
'/reference_generator/health_status', 
'/lidar_aft/detections_as_markers', 
'/lidar_fore/detections_as_markers', 
'/senti_parser/SentiTrigger',
'/crossing_behavior/stop_crossing',
'/sp_vp/sp_vp_log', 
'/sitaw/health_status', 
'/crossing_behavior/status_message', 
'/sp_vp/vis/interactive_obstacles/update', 
'/sp_vp/vis/obstacles', 
'/ravnkloa_radar_base_map', 
'/rgb/ap_a/image_raw/theora', 
'/navigation/health_status', 
'/lidar_aft/points', 
'/dp_interface/tau_manual', 
'/rgb/ap_a/image_raw', 
'/supervisory/select_autonomy_mode', 
'/crossing_behavior/status', 
'/tf', 
'/senti_parser/SentiIMU', 
'/dp_interface/health_status', 
'/radar/detector/points_statistics', 
'/lidar_aft/detector/points_statistics', 
'/io_machine/hatch_driver/hatch_command', 
'/sitaw/track_manager/fixed_rate_estimate_topic', 
'/crossing_behavior/go_to_dock', 
'/crossing_behavior/heartbeat', 
'/reference_generator/visualization', 
'/crossing_behavior/nominal_path_vis', 
'/crossing_behavior/accept_trajectory', 
'/sp_vp/mpcs_tracks', 
'/sp_vp/speed_reference_override/reference_speed', 
'/reference_generator/mode', 
'/io_machine/ultrasonic_dist_driver/ultrasonic_dist', 
'/supervisory/disable_autonomy', 
'/rosout', 
'/sp_vp/speed_reference_override/mode', 
'/sitaw/targets_as_informative_markers', 
'/reference_generator/waypoints', 
'/dp_interface/bms_status', 
'/rgb/ap_a/meta', 
'/rgb/ap_a/image_raw/compressot_reference', 
'/senti_parser/SentiTwist', 
'/crossing_behavior/moor_docked', 
'/senti_parser/SentiGNSSDop', 
'/radar/detections_as_markers', '/
"""