import numpy as np
import cv2
from RWPS import RWPS
import os
from utilities import pohang_2_intrinsics_extrinsics, mods_2_intrinsics_extrinsics, pohang_2_extract_roll_pitch_yaw, pohang_2_extract_camera_timstamps
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib
from stereo_cam import StereoCam
from fastSAM import FastSAMSeg
from stixels import Stixels
from stixels import *
from bev import calculate_bev_image
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from temporal_smoothing import TemporalSmoothing
from project_lidar_to_camera import *

#matplotlib.use('Agg')

dataset = "pohang"
start_frame = 0
save_video = False
show_horizon = False
create_bev = False
save_bev = False
create_polygon = False
plot_polygon = False
save_polygon_video = False
create_rectangular_stixels = True
use_temporal_smoothing = False
use_temporal_smoothing_ego_motion_compensation = True
visualize_ego_motion_compensation = True
mode = "fastsam" #"fastsam" #"rwps"
iou_threshold = 0.1
fastsam_model_path = "weights/FastSAM-x.pt"
device = "mps"
dataset_dir = "/Users/johannesskaro/Documents/KYB 5.år/Datasets/pohang"
onedrive_dir = "/Users/johannesskaro/OneDrive - NTNU/autumn-2023"
src_dir = "/Users/johannesskaro/Documents/KYB 5.år/fusedWSS"
sequence = "pohang00_pier" #"pohang00_port"


W, H = (2048, 1080)

basepath = f"{dataset_dir}/{sequence}/"
basepath_images = basepath + "stereo/"
left_image_path = f"{basepath_images}/left_images"
right_image_path = f"{basepath_images}/right_images"

unsorted_image_files = [
    f
    for f in os.listdir(left_image_path)
    if os.path.isfile(os.path.join(left_image_path, f))
]
image_files = sorted(unsorted_image_files, key=lambda x: int(x[:6]))
image_timestamps_file = basepath + "stereo/timestamp.txt"
image_timestamps = pohang_2_extract_camera_timstamps(image_timestamps_file)

extrinsics_file = basepath + "calibration/extrinsics.json"
intrinsics_file = basepath + "calibration/intrinsics.json"
intrinsics_json, extrinsics_json = ut.load_intrinsics_and_extrinsics(intrinsics_file, extrinsics_file)
intrinsics, extrinsics, t_ahrs_to_camera, R_ahrs_to_camera, R_ahrs_to_lidar, t_ahrs_to_lidar = pohang_2_intrinsics_extrinsics(intrinsics_json, extrinsics_json)

lidar_dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('time', np.uint32), ('reflectivity', np.uint16), ('ambient', np.uint16), ('range', np.uint32)]
lidar_data_path = f"{basepath}/lidar/lidar_front/points/"
unsorted_lidar_data = [
    f
    for f in os.listdir(lidar_data_path)
    if os.path.isfile(os.path.join(lidar_data_path, f))
]
lidar_data = sorted(unsorted_lidar_data, key=lambda x: int(x[:19]))

H = intrinsics["stereo_left"]["image_height"]
W = intrinsics["stereo_left"]["image_width"]


# import IMU data

ahrs_data_folder = (
    f"{basepath}/navigation/ahrs.txt"
)
ahrs_data_unmatched = pohang_2_extract_roll_pitch_yaw(ahrs_data_folder)
ahrs_data = ut.pohang_2_match_ahrs_timestamps(image_timestamps, ahrs_data_unmatched)

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_{mode}_{sequence}_temporal_smoothing_lidar.mp4",
        fourcc,
        10.0,
        (W, H),
    )

if save_polygon_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_polygon = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_polygon_{mode}_{sequence[:6]+sequence[-4:]}_temporal_smoothing.mp4",
        fourcc,
        10.0,
        (H, H),
    )

stereo = StereoCam(intrinsics, extrinsics)
cam_params = stereo.get_basic_camera_parameters()
P1 = stereo.get_left_projection_matrix()
h_fov, v_fov = stereo.get_fov()
KL = stereo.get_camera_matrix(cx=cam_params["cx"], cy=cam_params["cy"], fx=cam_params["fx"], fy=cam_params["fy"])

fastsam = FastSAMSeg(model_path=fastsam_model_path)
rwps3d = RWPS()
stixels = Stixels()
temporal_smoothing = TemporalSmoothing(5, KL, R_ahrs_to_camera, t_ahrs_to_camera)

config_rwps = f"{src_dir}/rwps_config.json"
p1 = (102, 22)
p2 = (102, 639)
invalid_depth_mask = rwps3d.set_invalid(p1, p2, shape=(H - 1, W))
rwps3d.set_camera_params(cam_params, P1)
rwps3d.set_config(config_rwps)

num_disparities = int(W * 0.08)
block_size = 4
stereo_matcher = cv2.StereoSGBM_create(
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


horizon_cutoff = H // 2
horizon_point0 = np.array([0, horizon_cutoff]).astype(int)
horizon_pointW = np.array([W, horizon_cutoff]).astype(int)

start_frame = 50
curr_frame = start_frame
num_frames = len(image_files)
horizon_msd_array = []
imu_data_array = []
iou_array = []

#print(image_timestamps)

while curr_frame < num_frames - 1:
    print(f"Frame {curr_frame} / {len(image_files)}")
    image_timestamp = image_timestamps[curr_frame][0]
    image_number = image_timestamps[curr_frame][1]
    ahrs_data_timestamp = ahrs_data[curr_frame][0]

    left_image_path = f"{basepath_images}/left_images/{image_number}.png"
    right_image_path = f"{basepath_images}/right_images/{image_number}.png"

    #print(f"Image timestamp: {image_timestamp}")
    #print(f"Ahrs data timestamp: {ahrs_data_timestamp}")


    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)
    orientation = ahrs_data[curr_frame][1:]

    scan_path = f"{lidar_data_path}/{lidar_data[curr_frame]}"
    scan = np.fromfile(scan_path, dtype=lidar_dtype)
    points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
    intensity = scan['intensity']
    lidar_points_c = transform_lidar_to_camera_frame(R_ahrs_to_lidar, t_ahrs_to_lidar, R_ahrs_to_camera, t_ahrs_to_camera, KL, points, intensity)
    #ut.visualize_lidar_points(points)

    disparity_img = stereo.get_disparity(
        left_img, right_img, rectify_images=True, stereo_matcher=stereo_matcher
    )
    depth_img = stereo.get_depth_map(
        left_img, right_img, rectify_images=True, stereo_matcher=stereo_matcher
    )
    depth_img[depth_img > 60] = 0

    ### COMMON
    (H, W, D) = left_img.shape

        # Run RWPS segmentation
    rwps_mask_3d, plane_params_3d, rwps_succeeded = rwps3d.segment_water_plane_using_point_cloud(
        left_img.astype(np.uint8), depth_img
    )
    rwps_mask_3d = rwps_mask_3d.astype(int)

    if horizon_cutoff >= H - 1:
        horizon_cutoff = H // 2

    water_mask = np.zeros_like(depth_img)

    # FastSAM classifier
    if mode == "fastsam":
        left_image_kept = left_img.copy()[horizon_cutoff:, :]
        rwps_mask_3d = rwps_mask_3d[horizon_cutoff:, :]

        water_mask = fastsam.get_mask_at_points(
            left_image_kept,
            [[(W - 1) // 2, left_image_kept.shape[0] - 100]],
            pointlabel=[1],
            device=device,
        ).astype(int)
        water_mask = water_mask.reshape(rwps_mask_3d.shape)

        water_mask2 = np.zeros((H, W), dtype=np.uint8)
        water_mask2[horizon_cutoff:] = water_mask
        water_mask = water_mask2

    # Run FastSAM segmentation
    elif mode == "fusion":

        if rwps_succeeded == False:
            #Use fastsam
            print("RWPS failed, using FastSAM")
            left_image_kept = left_img.copy()[horizon_cutoff:, :]
            rwps_mask_3d = rwps_mask_3d[horizon_cutoff:, :]

            water_mask = fastsam.get_mask_at_points(
                left_image_kept,
                [[(W - 1) // 2, left_image_kept.shape[0] - 100]],
                pointlabel=[1],
                device=device,
            ).astype(int)
            water_mask = water_mask.reshape(rwps_mask_3d.shape)

            water_mask2 = np.zeros((H, W), dtype=np.uint8)
            water_mask2[horizon_cutoff:] = water_mask
            water_mask = water_mask2

        else:
            print("RWPS succeeded, using fusion")
            left_img_cut = left_img.copy()[horizon_cutoff:, :]
            fastsam_masks_cut = fastsam.get_all_masks(left_img_cut, device=device).astype(
                int
            )
            fastsam_masks = np.full((fastsam_masks_cut.shape[0], H, W), False)

            if len(fastsam_masks):
                fastsam_masks[:, horizon_cutoff:, :] = fastsam_masks_cut

                # Stack all FastSAM masks into a 3D array (height, width, num_masks)
                fastsam_masks_stack = np.stack(fastsam_masks, axis=-1)

                # Calculate IoU for each mask in a vectorized manner
                iou_scores = np.array(
                    [
                        ut.calculate_iou(rwps_mask_3d, fastsam_masks_stack[:, :, i])
                        for i in range(fastsam_masks_stack.shape[-1])
                    ]
                )

                # Find the index of the mask with maximum IoU
                max_corr_index = np.argmax(iou_scores)
                keep_indexes = np.argwhere(iou_scores > iou_threshold)

                # Combine the selected FastSAM mask with the rwps mask
                water_mask = np.clip(
                    fastsam_masks_stack[:, :, max_corr_index] + rwps_mask_3d, 0, 1
                )

                # Create a mask indicating which masks to subtract (all masks except the one with max correlation)
                masks_to_subtract = np.ones(fastsam_masks.shape[0], dtype=bool)
                masks_to_subtract[keep_indexes] = False

                # Subtract all the remaining FastSAM masks from water_mask in a vectorized manner
                distractions_mask = np.any(fastsam_masks[masks_to_subtract], axis=0)
                water_mask = np.clip(water_mask - distractions_mask, 0, 1)
                iou_subtracts = iou_scores[masks_to_subtract]
                #print(
                #    iou_scores[keep_indexes],
                #    np.sort(iou_subtracts[iou_subtracts != 0])[::-1],
                #)

    else:
        water_mask = rwps_mask_3d
        distractions_mask = np.zeros_like(water_mask)

    if use_temporal_smoothing:
        water_mask = temporal_smoothing.get_smoothed_water_mask(water_mask)

    if use_temporal_smoothing_ego_motion_compensation:
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        water_mask_raw = water_mask
        water_mask = temporal_smoothing.get_smoothed_ego_motion_compensated_mask(
            water_mask_raw, roll, pitch, yaw
        )


        past_N_compensated_masks = temporal_smoothing.get_ego_motion_compensated_masks(
            roll, pitch, yaw
        )
        if visualize_ego_motion_compensation:
            temporal_smoothing.plot_overlay_compensated_vs_noncompensated(past_N_compensated_masks, water_mask_raw)

        
    blue_water_mask = water_mask

    nonwater_mask = np.logical_not(water_mask)
    nonwater_contrastreduced = left_img.copy()
    #nonwater_contrastreduced[nonwater_mask] = (
    #    nonwater_contrastreduced[nonwater_mask] // 2
    #) + 128

    water_img = nonwater_contrastreduced.copy()
    water_img = ut.blend_image_with_mask(
        water_img, blue_water_mask, [255, 100, 0], alpha1=1, alpha2=0.5
    )

    lidar_water_mask_image = merge_lidar_onto_image(water_img, lidar_points_c, intensity)

    if show_horizon:
        cv2.line(
            water_img,
            horizon_point0.astype(int),
            horizon_pointW.astype(int),
            [57, 255, 20],
            thickness=5,
        )

    if create_rectangular_stixels:
        rectangular_stixel_mask = stixels.create_rectangular_stixels(water_mask, disparity_img)
        cv2.imshow("Rectangular Stixels", rectangular_stixel_mask.astype(np.uint8) * 255)


    if create_polygon:    
        stixel_mask, stixel_positions = stixels.get_stixels_base(water_mask)
        stixel_width = stixels.get_stixel_width(W)
        stixels_2d_points = calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_img, cam_params)
        stixels_polygon = create_polygon_from_2d_points(stixels_2d_points)

        cv2.imshow("Stixels", stixel_mask.astype(np.uint8) * 255)

    if plot_polygon:

        myPoly = gpd.GeoSeries([stixels_polygon])
        myPoly.plot()
        plt.draw()
        plt.pause(0.5)
        plt.close()

    if save_polygon_video:

        myPoly = gpd.GeoSeries([stixels_polygon])
        dpi = 100
        fig, ax = plt.subplots(figsize=((H) / dpi, (H ) / dpi), dpi=dpi)
        myPoly.plot(ax=ax)
        fig.canvas.draw()
        # Convert the plot to a numpy array (RGB image)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to (height, width, 4)
        plt.close(fig)
        img_rgb = img[:, :, :3]

        polygon_BEV_image_resized = cv2.resize(img_rgb, (H, H))
        polygon_BEV_image_resized_bgr = cv2.cvtColor(polygon_BEV_image_resized, cv2.COLOR_RGB2BGR)
        out_polygon.write(polygon_BEV_image_resized_bgr)

    if save_video:
        #out.write(water_img)
        out.write(lidar_water_mask_image)
    #cv2.imshow("Water Segmentation", water_img)
    cv2.imshow("Depth", depth_img.astype(np.uint8))
    cv2.imshow("Lidar Water Mask", lidar_water_mask_image)
    key = cv2.waitKey(10)

    if key == 27:  # Press ESC to exit
        break
    if key in [106]:  # j
        curr_frame += 50
    elif key in [98]:  # b
        curr_frame -= 50
    elif key in [107]:  # k
        curr_frame += 200
    elif key in [110]:  # n
        curr_frame -= 200

    curr_frame += 1

cv2.destroyAllWindows()
if save_video:
    out.release()
if save_polygon_video:
    out_polygon.release()

    