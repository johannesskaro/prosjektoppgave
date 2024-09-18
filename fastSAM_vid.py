import numpy as np
from ultralytics import FastSAM
import cv2
import os

# Load model
model_path = "weights/FastSAM-s.pt"
model = FastSAM(model_path)
device = "mps"

# Load one image
img_path = "data/00022351L.jpg"
img = cv2.imread(img_path)
imgsz = img.shape
width, height = imgsz[1], imgsz[0]

# Load video
fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # You can also use 'MP4V' for .mp4 format
video = cv2.VideoWriter('segmented_video_fastSAM-s.mp4', fourcc, 10.0, (width, height))

basepath_images = "/Users/johannesskaro/Documents/KYB 5.år/Datasets/MODD2/video_data/kope81-00-00019370-00019710/framesRectified/"
unsorted_image_files = [
        f
        for f in os.listdir(basepath_images)
        if os.path.isfile(os.path.join(basepath_images, f))]
image_files = sorted(unsorted_image_files, key=lambda x: int(x[:8]))

start_frame = 0
# for curr_frame in range(start_frame, len(image_files), 2):
curr_frame = start_frame
num_frames = len(image_files)

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

    print(f"Frame {curr_frame//2} / {len(image_files)//2}")
    # Rectified
    #left_img = cv2.imread(left_image_path)
    #right_img = cv2.imread(right_image_path)

        # Check if images were loaded successfully
    #if left_img is None or right_img is None:
    #    print(f"Failed to load images: {left_image_path} or {right_image_path}")
    #    break
    
    point_results = model(left_image_path, points=[(width-1)/2, height-200], device=device, retina_masks=True, verbose=False) #point prompt bottom center pixel

    seg_mat = point_results[0].plot()

    video.write(seg_mat)
    curr_frame += 2
 
# Release the video writer object
video.release()
cv2.destroyAllWindows()

