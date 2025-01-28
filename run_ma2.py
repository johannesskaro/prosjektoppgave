import open3d as o3d
import numpy as np
from scipy.io import loadmat
from stixels import *
from RWPS import RWPS
from fastSAM import FastSAMSeg
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilities_ma2 import *
from project_lidar_to_camera import merge_lidar_onto_image
import utilities as ut
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib
from temporal_smoothing import TemporalSmoothing
import json
from run_3d_viz import plot_scene, animate
import os
from plotting import *

#matplotlib.use('Agg')

save_video = False
show_horizon = False
create_bev = False
save_bev = False
create_polygon = False
plot_polygon = False
save_polygon_video = False
create_rectangular_stixels = True
use_temporal_smoothing = True
use_temporal_smoothing_ego_motion_compensation = False
visualize_ego_motion_compensation = False
save_3d_stixels = False
save_3d_visualization_video = False

dataset = "ma2"
sequence = "scen6"
mode = "fusion" #"fastsam" #"rwps" #"fusion"
iou_threshold = 0.1
fastsam_model_path = "weights/FastSAM-x.pt"
device = "cuda"
src_dir = "/home/johannes/Documents/blueboats/prosjektoppgave"

W, H = (1920, 1080)
FPS = 15.0

plt.ion()

if save_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_{mode}_{sequence}_only_stixels.mp4",
        fourcc,
        FPS,
        (W, H),
    )

if save_polygon_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_polygon = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_{mode}_{sequence}_polygon_BEV.mp4",
        fourcc,
        FPS,
        (H, H),
    )

if save_3d_visualization_video:
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
    out_3d_visualization = cv2.VideoWriter(
        f"{src_dir}/results/video_{dataset}_{mode}_{sequence}_3d_visualization_v3.mp4",
        fourcc,
        FPS,
        (1280, 720),
    )

fastsam = FastSAMSeg(model_path=fastsam_model_path)
rwps3d = RWPS()
stixels = Stixels()
temporal_smoothing = TemporalSmoothing(5, K)
 
cam_params = {"cx": K[0,2], "cy": K[1,2], "fx": K[0,0], "fy":K[1,1], "b": baseline}
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

config_rwps = f"{src_dir}/rwps_config.json"
p1 = (102, 22)
p2 = (102, 639)
invalid_depth_mask = rwps3d.set_invalid(p1, p2, shape=(H - 1, W))
rwps3d.set_camera_params(cam_params, P1)
rwps3d.set_config(config_rwps)

horizon_cutoff = H // 2
horizon_point0 = np.array([0, horizon_cutoff]).astype(int)
horizon_pointW = np.array([W, horizon_cutoff]).astype(int)


def main():

    gen_lidar = gen_ma2_lidar_points()
    gen_svo = gen_svo_images()

    next_ma2_timestamp, next_ma2_lidar_points, intensity, xyz_c = next(gen_lidar)
    next_svo_timestamp, next_svo_image, disparity_img, depth_img = next(gen_svo)
    current_timestamp = 0

    iterating = True
    curr_frame = 0

    while iterating:
        if next_ma2_timestamp is not None and next_svo_timestamp is not None:
            if next_ma2_timestamp < next_svo_timestamp:
                #print("Lidar")
                try:
                    next_ma2_timestamp, next_ma2_lidar_points, intensity, xyz_c = next(gen_lidar)
                    current_timestamp = next_ma2_timestamp
                except StopIteration:
                    iterating = False
                    next_ma2_timestamp = None
                    next_ma2_lidar_points = None
            else:
                #print("SVO")
                #cv2.imshow("SVO image", next_svo_image)
                try:
                    next_svo_timestamp, next_svo_image, disparity_img, depth_img = next(gen_svo)
                    current_timestamp = next_svo_timestamp
                except StopIteration:
                    iterating = False
                    next_svo_timestamp = None
                    next_svo_image = None
        elif next_ma2_timestamp is not None:
            #print("Lidar")
            try:
                next_ma2_timestamp, next_ma2_lidar_points, intensity, xyz_c = next(gen_lidar)
                current_timestamp = next_ma2_timestamp
            except StopIteration:
                iterating = False
                next_ma2_timestamp = None
                next_ma2_lidar_points = None
        elif next_svo_timestamp is not None:
            #print("SVO")
            #cv2.imshow("SVO image", next_svo_image)
            try:
                next_svo_timestamp, next_svo_image, disparity_img, depth_img = next(gen_svo)
                current_timestamp = next_svo_timestamp
            except StopIteration:
                iterating = False
                next_svo_timestamp = None
                next_svo_image = None
        else:
            break
        
        print(f"Current timestamp: {current_timestamp}")
        curr_frame += 1

        left_img = next_svo_image
        lidar_image_points = np.squeeze(next_ma2_lidar_points, axis=1)  # From (N, 1, 2) to (N, 2)
        lidar_3d_points = xyz_c

        (H, W, D) = left_img.shape

        # Run RWPS segmentation
        rwps_mask_3d, plane_params_3d, rwps_succeded = rwps3d.segment_water_plane_using_point_cloud(
            left_img.astype(np.uint8), depth_img
        )
        rwps_mask_3d = rwps_mask_3d.astype(int)

        water_mask = np.zeros_like(depth_img)

        # FastSAM classifier
        if mode == "fastsam":
            #left_image_kept = left_img.copy()[horizon_cutoff:, :]
            #rwps_mask_3d = rwps_mask_3d[horizon_cutoff:, :]

            left_image_kept = left_img.copy()

            water_mask = fastsam.get_mask_at_points(
                left_image_kept,
                [[(W - 1) // 2, left_image_kept.shape[0] - 100]],
                pointlabel=[1],
                device=device,
            ).astype(int)
            water_mask = water_mask.reshape(rwps_mask_3d.shape)

            #water_mask2 = np.zeros((H, W), dtype=np.uint8)
            #water_mask2[horizon_cutoff:] = water_mask
            #water_mask = water_mask2

        # Run FastSAM segmentation
        elif mode == "fusion":

            if rwps_succeded == False:
                #Use fastsam
                print("RWPS failed, using FastSAM")
                #left_image_kept = left_img.copy()[horizon_cutoff:, :]
                #rwps_mask_3d = rwps_mask_3d[horizon_cutoff:, :]

                left_image_kept = left_img.copy()

                water_mask = fastsam.get_mask_at_points(
                    left_image_kept,
                    [[(W - 1) // 2, left_image_kept.shape[0] - 100]],
                    pointlabel=[1],
                    device=device,
                ).astype(int)
                water_mask = water_mask.reshape(rwps_mask_3d.shape)

                #water_mask2 = np.zeros((H, W), dtype=np.uint8)
                #water_mask2[horizon_cutoff:] = water_mask
                #water_mask = water_mask2

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
        
        blue_water_mask = water_mask

        nonwater_mask = np.logical_not(water_mask)
        nonwater_contrastreduced = left_img.copy()
        nonwater_contrastreduced[nonwater_mask] = (
            nonwater_contrastreduced[nonwater_mask] // 2
        ) + 128

        left_img_contrastreduced = left_img.copy() // 2 + 128
        blue_color = [255, 100, 0] 
        pink_color = [255, 0, 255]
        water_img = nonwater_contrastreduced.copy()
        water_img = ut.blend_image_with_mask(
            water_img, blue_water_mask, pink_color, alpha1=1, alpha2=0.5
        )

        if create_rectangular_stixels:
            rec_stixel_list, rec_stixel_mask = stixels.create_rectangular_stixels(water_mask, disparity_img, depth_img)
            #cv2.imshow("Rectangular Stixels", rectangular_stixel_mask.astype(np.uint8) * 255)
            _, free_space_boundary = stixels.get_free_space_boundary(water_mask)

            stixel_mask, stixel_positions = stixels.get_stixels_base(water_mask)

            filtered_lidar_points, filtered_lidar_3d_points, lidar_stixel_indices = stixels.filter_lidar_points_by_stixels(lidar_image_points, lidar_3d_points)
            lidar_stixel_depths = stixels.get_stixel_depth_from_lidar_points(filtered_lidar_3d_points, lidar_stixel_indices)
            stixels_2d_points = stixels.get_polygon_points_from_lidar_and_stereo_depth(lidar_stixel_depths, stixel_positions, cam_params)
            stixels_polygon = create_polygon_from_2d_points(stixels_2d_points)
            #stixels_3d_points = stixels.get_stixel_3d_points(cam_params)



        if create_polygon:    
            stixel_mask, stixel_positions = stixels.get_stixels_base(water_mask)
            stixel_width = stixels.get_stixel_width(W)
            stixels_2d_points = stixels.calculate_2d_points_from_stixel_positions(stixel_positions, stixel_width, depth_img, cam_params)
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
            ax.set_xlim(-30, 30)
            ax.set_ylim(0, 60)
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

        if save_3d_stixels:
            data = {
                "stixels": stixels_3d_points.tolist(),
                "plane_params": plane_params_3d.tolist(),
                "water_surface_polygon_points": stixels_2d_points.tolist(),
            }
            with open(f"files/temp.json", "w") as f:
                json.dump(data, f)
            os.replace("files/temp.json", "files/stixel.json")

        if save_3d_visualization_video:
            stixels_3d_points = stixels.get_stixel_3d_points(cam_params)
            dpi = 100
            fig = plt.figure(figsize=(1280/dpi, 720/dpi), dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('persp', focal_length=0.2) 
            azim = 0 + 3 * np.sin(np.radians(curr_frame * 2))  # Azimuth changes within a 10-15 degree range
            elev = 20 + 10 * np.sin(np.radians(curr_frame * 1))  # Elevation changes within a 13-17 degree range
            camera_position = (elev, azim)
            plot_scene(ax, stixels_3d_points.tolist(), stixels_2d_points, plane_params_3d, camera_position)
                
            # Save current frame to a file
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            img = img[:, :, :3]

            img = cv2.resize(img, (1280, 720))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            out_3d_visualization.write(img)
            
 


        image_with_stixels = stixels.merge_stixels_onto_image(rec_stixel_list, left_img)
        #image_with_stixels_2 = stixels.merge_stixels_onto_image(rec_stixel_list, left_img_contrastreduced)
        image_with_stixels_and_filtered_lidar = merge_lidar_onto_image(image_with_stixels, filtered_lidar_points)

        

        #water_img_with_free_space_boundary = ut.blend_image_with_mask(
        #    image_with_stixels_and_filtered_lidar, free_space_boundary, [0, 255, 255], alpha1=1, alpha2=1
        #)
        #cv2.imwrite("files/water_img_with_free_space_boundary.png", water_img_with_free_space_boundary)
        #cv2.imshow("Water Segmentation", water_img_with_free_space_boundary)
        #cv2.imwrite("files/water_mask.png", water_mask.astype(np.uint8) * 255)
        #cv2.imwrite("files/free_space_boundary_mask.png", free_space_boundary.astype(np.uint8) * 255)

        #image_with_stixels_and_free_space_boundary = ut.blend_image_with_mask(
        #   image_with_stixels_2, free_space_boundary, [0, 255, 255], alpha1=1, alpha2=1
        #)
        #plot_disparity_column(disparity_img, rec_stixel_mask, 610)
        #plot_stixel_img_without_column(image_with_stixels_and_free_space_boundary, rec_stixel_mask, 610)


        #image_with_lidar = merge_lidar_onto_image(left_img, lidar_image_points)

        cv2.imshow("Left image", left_img)
        
        #cv2.imshow("Image with lidar", image_with_lidar)
        #cv2.imwrite("files/image_with_lidar.png", image_with_lidar)
        cv2.imshow("Stixel image", image_with_stixels_and_filtered_lidar)
        #cv2.imwrite("files/stixel_image.png", image_with_stixels_and_filtered_lidar)
        #cv2.imshow("Water segmentation", water_img_with_free_space_boundary)
        #cv2.imshow("Depth image", depth_img)
        #cv2.imshow("Left image", left_img)
        #rec_stixel_mask *= 255
        #cv2.imshow("Stixel mask", rec_stixel_mask.astype(np.uint8))




        if save_video:
            #out.write(water_img)
            out.write(image_with_stixels_and_filtered_lidar)
            #out.write(water_img_with_free_space_boundary)

        key = cv2.waitKey(10)

        if key == 27:  # Press ESC to exit
            break
        if key in [106]:  # j
            data = {
                "stixels": stixels_3d_points.tolist(),
                "plane_params": plane_params_3d.tolist(),
                "water_surface_polygon_points": stixels_2d_points.tolist(),
            }
            with open(f"files/temp.json", "w") as f:
                json.dump(data, f)
            os.replace("files/temp.json", "files/stixel.json")
            print("Saved stixels")

        cv2.waitKey(10)

    cv2.destroyAllWindows()
    if save_video:
        out.release()
    if save_polygon_video:
        out_polygon.release()
    if save_3d_visualization_video:
        out_3d_visualization.release()


if __name__ == "__main__":
    main()