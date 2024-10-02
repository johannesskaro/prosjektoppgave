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
from bev import calculate_bev_image
from stixels import Stixels
from stixels import create_polygon_from_3d_points
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt

matplotlib.use('Agg')

dataset = "summer_2023"
start_frame = 0
save_video = False
show_horizon = False
create_bev = False
save_bev = False
plot_polygon = False
save_polygon_video = True
mode = "fusion" #"rwps"
iou_threshold = 0.1
fastsam_model_path = "weights/FastSAM-x.pt"
device = "mps"
dataset_dir = "/Users/johannesskaro/Documents/KYB 5.år/Datasets"
onedrive_dir = "/Users/johannesskaro/OneDrive - NTNU/summer-2023"
src_dir = "/Users/johannesskaro/Documents/KYB 5.år/fusedWSS"


#sequence = "2023-07-11_11-46-15_5256916_HD1080_FPS15"
sequence = "2023-07-11_12-49-30_5256916_HD1080_FPS15"

RESULTS_FOLDER = f"{onedrive_dir}/{sequence}"

W, H = (1920, 1080)
FPS = 15.0
K = np.loadtxt(f"{RESULTS_FOLDER}/left/K_matrix.txt")
R = np.loadtxt(f"{RESULTS_FOLDER}/left/R_matrix.txt")
T = np.loadtxt(f"{RESULTS_FOLDER}/left/T_matrix.txt")

plt.ion()

left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULTS_FOLDER}/left")))
timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
timestamps = sorted(timestamps)

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_fusedWSS_boat.mp4",
        fourcc,
        FPS,
        (W, H),
    )

if save_polygon_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_polygon = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_polygon_BEV_wide_stixels_dock.mp4",
        fourcc,
        FPS,
        (H, H),
    )

fastsam = FastSAMSeg(model_path=fastsam_model_path)
rwps3d = RWPS()
stixels = Stixels()

baseline = np.linalg.norm(T)
cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy":K[1,1], "b": baseline}
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

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
num_frames = len(timestamps)
horizon_msd_array = []
imu_data_array = []
iou_array = []

for ti in range(0, len(timestamps)):
    timestamp = timestamps[ti]
    left_img = cv2.imread(f"{RESULTS_FOLDER}/left/{timestamp}.png")
    right_img = cv2.imread(f"{RESULTS_FOLDER}/right/{timestamp}.png")
    disparity_img = np.array(cv2.imread(f"{RESULTS_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)

    print(f"Frame {ti} / {num_frames}")

    f = cam_params["fx"]

    depth_img = baseline * f / disparity_img

    depth_img[depth_img > 40] = 0

    ### COMMON
    (H, W, D) = left_img.shape

    # Run RWPS segmentation
    rwps_mask_3d, plane_params_3d, rwps_succeded = rwps3d.segment_water_plane_using_point_cloud(
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
            [[(W - 1) // 2, left_image_kept.shape[0] - 1]],
            pointlabel=[1],
            device=device,
        ).astype(int)
        water_mask = water_mask.reshape(rwps_mask_3d.shape)

        water_mask2 = np.zeros((H, W), dtype=np.uint8)
        water_mask2[horizon_cutoff:] = water_mask
        water_mask = water_mask2

    # Run FastSAM segmentation
    elif mode == "fusion":

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

    stixel_mask, stixel_positions = stixels.get_stixels(water_mask)
    stixel_width = stixels.get_stixel_width(W)
    stixels_3d_points = ut.calculate_3d_points_from_stixel_positions(stixel_positions, stixel_width, depth_img, cam_params)
    stixels_polygon = create_polygon_from_3d_points(stixels_3d_points)

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
        fig, ax = plt.subplots(figsize=(H / dpi, H / dpi), dpi=dpi)
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



    if create_bev:
        bev_image = None
        points_ccf = ut.calculate_3d_points_from_mask(water_mask, depth_img, cam_params)
        if mode == "fusion":
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

        elif mode == "fastsam" or mode == "rwps":
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

