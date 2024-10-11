import numpy as np
import cv2
from RWPS import RWPS
import os
from utilities import pohang_2_intrinsics_extrinsics, mods_2_intrinsics_extrinsics
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib
from stereo_cam import StereoCam
from fastSAM import FastSAMSeg
from stixels import Stixels
from stixels import create_polygon_from_2d_points
from bev import calculate_bev_image
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from temporal_smoothing import TemporalSmoothing

#matplotlib.use('Agg')

dataset = "modd2"
start_frame = 0
save_video = False
show_horizon = False
create_bev = False
save_bev = False
create_polygon = False
plot_polygon = False
save_polygon_video = False
use_temporal_smoothing = False
mode = "fusion" #"fastsam" #"rwps"
iou_threshold = 0.1
fastsam_model_path = "weights/FastSAM-x.pt"
device = "mps"
dataset_dir = "/Users/johannesskaro/Documents/KYB 5.år/Datasets"
onedrive_dir = "/Users/johannesskaro/OneDrive - NTNU/autumn-2023"
src_dir = "/Users/johannesskaro/Documents/KYB 5.år/fusedWSS"

if dataset == "modd2":
    #sequence = "kope81-00-00006800-00007095"
    #sequence = "kope81-00-00019370-00019710" #cruise ship
    #sequence = "kope75-00-00013780-00014195" #gummibåt
    #sequence = "kope75-00-00062200-00062500" #havn
    #sequence = "kope71-01-00014337-00014547" #good performance
    #sequence = "kope75-00-00037550-00037860" # good performance
    sequence = "kope75-00-00021500-00022160"
    

    W, H = (1278, 958)

    basepath = f"{dataset_dir}/MODD2/video_data/{sequence}/"
    basepath_images = basepath + "framesRectified/"
    basepath_gt = f"{dataset_dir}/MODD2/annotationsV2_rectified/{sequence}/ground_truth/"

    unsorted_image_files = [
        f
        for f in os.listdir(basepath_images)
        if os.path.isfile(os.path.join(basepath_images, f))
    ]
    image_files = sorted(unsorted_image_files, key=lambda x: int(x[:8]))
    config_file = basepath + "calibration.yaml"
    fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
    M1 = fs.getNode("M1").mat()
    M2 = fs.getNode("M2").mat()
    D1 = fs.getNode("D1").mat()
    D2 = fs.getNode("D2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    intrinsics, extrinsics = mods_2_intrinsics_extrinsics(M1, M2, D1, D2, R, T, (W, H))
    H = intrinsics["stereo_left"]["image_height"]
    W = intrinsics["stereo_left"]["image_width"]
    usvparts_path = (
        f"{dataset_dir}/MODD2/USV_parts_masks/{sequence[:6]}"
    )
    mask_not_usvpartsL = np.logical_not(plt.imread(f"{usvparts_path}_L.png"))[1:, :]
    mask_not_usvpartsL_points = [[0, 0], [0, 0], [0, 0], [0, 0]]
    if sequence[:6] == "kope81":
        mask_not_usvpartsL = np.full((H - 1, W), fill_value=True, dtype=bool)
        mask_not_usvpartsL_points = [[957, 900], [1024, 957]]
        mask_not_usvpartsL[900:, 957:1024] = False

    # import IMU data
    imu_data_folder = (
        f"{dataset_dir}/MODD2/video_data/{sequence}/imu"
    )
    unsorted_imu_files = [
        f
        for f in os.listdir(imu_data_folder)
        if os.path.isfile(os.path.join(imu_data_folder, f))
    ]

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_{mode}_{sequence[:6]+sequence[-4:]}_temporal_smoothing.mp4",
        fourcc,
        10.0,
        (W, H-1),
    )

if save_polygon_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_polygon = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_polygon_{mode}_{sequence[:6]+sequence[-4:]}_temporal_smoothing.mp4",
        fourcc,
        10.0,
        (H-1, H-1),
    )

stereo = StereoCam(intrinsics, extrinsics)
fastsam = FastSAMSeg(model_path=fastsam_model_path)
rwps3d = RWPS()
stixels = Stixels()
temporal_smoothing = TemporalSmoothing()

stereo.plot_BEV_depth_uncertainty()

cam_params = stereo.get_basic_camera_parameters()
P1 = stereo.get_left_projection_matrix()
h_fov, v_fov = stereo.get_fov()

config_rwps = f"{src_dir}/rwps_config.json"
p1 = (102, 22)
p2 = (102, 639)
invalid_depth_mask = rwps3d.set_invalid(p1, p2, shape=(H - 1, W))
rwps3d.set_camera_params(cam_params, P1)
rwps3d.set_config(config_rwps)

roll_values3d = []
pitch_values3d = []
gt_point0andWarray = []

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

start_frame = 0
curr_frame = start_frame
num_frames = len(image_files)
horizon_msd_array = []
imu_data_array = []
iou_array = []

while curr_frame < num_frames - 1:
    assert image_files[curr_frame][:8] == image_files[curr_frame + 1][:8]
    if (
        image_files[curr_frame][8:] == "L.jpg"
        and image_files[curr_frame + 1][8:] == "R.jpg"
    ):
        left_image_path = basepath_images + image_files[curr_frame]
        right_image_path = basepath_images + image_files[curr_frame + 1]
    elif (
        image_files[curr_frame][8:] == "R.jpg"
        and image_files[curr_frame + 1][8:] == "L.jpg"
    ):
        left_image_path = basepath_images + image_files[curr_frame + 1]
        right_image_path = basepath_images + image_files[curr_frame]
    else:
        print("Mismatched images")
        break

    imu_path = f"{imu_data_folder}/{image_files[curr_frame][:8]}.txt"

    print(f"Frame {curr_frame//2} / {len(image_files)//2}")

    # Rectified
    left_img = cv2.imread(left_image_path)[1:, :]
    right_img = cv2.imread(right_image_path)[1:, :]
    imu_data = np.loadtxt(imu_path)
    imu_data_array.append(np.loadtxt(imu_path))

    gt_path = basepath_gt + image_files[curr_frame][:8] + "L.mat"
    gt_ann = ut.read_mat_file(gt_path)["annotations"]

    assert gt_ann.shape[0] == 1
    assert gt_ann.shape[1] == 1

    gt_types = gt_ann[0].dtype
    gt_waterline = gt_ann[0][0][0]
    gt_obstacles = gt_ann[0][0][1]

    sea_edge_x = [p[0] for p in gt_waterline]
    sea_edge_y = [p[1] for p in gt_waterline]

    disparity_img = stereo.get_disparity(
        left_img, right_img, rectify_images=False, stereo_matcher=stereo_matcher
    )
    depth_img = stereo.get_depth_map(
        left_img, right_img, rectify_images=False, stereo_matcher=stereo_matcher
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

    if rwps3d.prev_planemodel_disp is not None:
        roll = rwps3d.get_roll()
        pitch = rwps3d.get_pitch()
        roll_values3d.append(np.rad2deg(roll))
        pitch_values3d.append(np.rad2deg(pitch))
        roll_values3d.append(np.rad2deg(rwps3d.get_roll()))
        pitch_values3d.append(np.rad2deg(rwps3d.get_pitch()))
        p_disp = rwps3d.prev_planemodel_disp[:3] / np.linalg.norm(rwps3d.prev_planemodel_disp[:3])
        horizon_point0, horizon_pointW, horizon_cutoff = rwps3d.get_horizon_from_disparity_plane_parameters(p_disp)

        print(
            f"Pitch: {np.round(np.rad2deg(pitch),2)}, Roll: {np.round(np.rad2deg(roll),2)}, P1: {horizon_point0}, P2: {horizon_pointW}"
        )

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


    water_mask = np.logical_and(water_mask, mask_not_usvpartsL)
    #water_mask = ut.remove_obstacles_from_watermask(water_mask, gt_obstacles)
    if use_temporal_smoothing:
        water_mask = temporal_smoothing.get_smoothed_water_mask(water_mask)

    blue_water_mask = water_mask

    nonwater_mask = np.logical_not(water_mask)
    nonwater_contrastreduced = left_img.copy()
    nonwater_contrastreduced[nonwater_mask] = (
        nonwater_contrastreduced[nonwater_mask] // 2
    ) + 128

    water_img = nonwater_contrastreduced.copy()
    water_img = ut.blend_image_with_mask(
        water_img, blue_water_mask, [255, 100, 0], alpha1=1, alpha2=0.5
    )

    if show_horizon:
        cv2.line(
            water_img,
            horizon_point0.astype(int),
            horizon_pointW.astype(int),
            [57, 255, 20],
            thickness=5,
        )


    if create_polygon:    
        stixel_mask, stixel_positions = stixels.get_stixels(water_mask)
        stixel_width = stixels.get_stixel_width(W)
        stixels_2d_points = ut.calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_img, cam_params)
        stixels_polygon = create_polygon_from_2d_points(stixels_2d_points)

        cv2.imshow("Stixels", stixel_mask.astype(np.uint8) * 255)
        cv2.imshow("Depth", depth_img.astype(np.uint8) * 255)

    if plot_polygon:

        myPoly = gpd.GeoSeries([stixels_polygon])
        myPoly.plot()
        plt.draw()
        plt.pause(0.5)
        plt.close()

    if save_polygon_video:

        myPoly = gpd.GeoSeries([stixels_polygon])
        dpi = 100
        fig, ax = plt.subplots(figsize=((H-1) / dpi, (H -1 ) / dpi), dpi=dpi)
        myPoly.plot(ax=ax)
        fig.canvas.draw()
        # Convert the plot to a numpy array (RGB image)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Reshape to (height, width, 4)
        plt.close(fig)
        img_rgb = img[:, :, :3]

        polygon_BEV_image_resized = cv2.resize(img_rgb, (H-1, H-1))
        polygon_BEV_image_resized_bgr = cv2.cvtColor(polygon_BEV_image_resized, cv2.COLOR_RGB2BGR)
        out_polygon.write(polygon_BEV_image_resized_bgr)

    if create_bev:
        bev_image = None
        #points_ccf = ut.calculate_3d_points_from_mask(water_mask, depth_img, cam_params)
        points_ccf = ut.calculate_3d_points_from_mask(stixel_mask, depth_img, cam_params)
        if mode == "fusion" and rwps_succeeded:
            subtract_points_ccf = ut.calculate_3d_points_from_mask(
                distractions_mask, depth_img, cam_params
            )
            points = np.vstack([points_ccf, subtract_points_ccf])
            colors = np.vstack(
                [
                    np.repeat([[200, 100, 0]], points_ccf.shape[0], axis=0),
                    np.repeat([[0, 0, 255]], subtract_points_ccf.shape[0], axis=0),
                ]
            )  # BRG Blue/red
            bev_image = calculate_bev_image(
                points, colors=colors, image_size=(500, 600)
            )

        else:
            colors = np.array(points_ccf.shape[0] * [[200, 100, 0]])
            bev_image = calculate_bev_image(
                points_ccf, colors=colors, image_size=(500, 600)
            )

        cv2.imshow("BEV", bev_image)
    
    if save_video:
        out.write(water_img)
    cv2.imshow("Water Segmentation", water_img)
    # cv2.imshow("Depth", depth_img)
    key = cv2.waitKey(10)

    if key == 27:  # Press ESC to exit
        break
    if key in [83, 115]:  # Press s or S to save image
        cv2.imwrite(
            f"{src_dir}/results/{dataset}_{mode}_{sequence[:6]+sequence[-4:]}_frame{curr_frame}.png",
            water_img,
        )
        if ut.save_bev and create_bev:
            cv2.imwrite(
                f"{src_dir}/results/{dataset}_{mode}_{sequence[:6]+sequence[-4:]}_frame{curr_frame}_BEV.png",
                bev_image,
            )
        print(f"Saved frame {curr_frame}. Saved BEV: {save_bev}")
    if key in [106]:  # j
        curr_frame += 50
    elif key in [98]:  # b
        curr_frame -= 50
    elif key in [107]:  # k
        curr_frame += 200
    elif key in [110]:  # n
        curr_frame -= 200

    curr_frame += 2

cv2.destroyAllWindows()
if mode == "horizon":
    plt.plot(roll_values3d)
    plt.plot(pitch_values3d)
    plt.show()
if save_video:
    out.release()
if save_polygon_video:
    out_polygon.release()

