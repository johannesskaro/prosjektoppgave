import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

sequence = "2023-07-11_11-46-15_5256916_HD1080_FPS15"
#sequence = "2023-07-11_12-49-30_5256916_HD1080_FPS15"
onedrive_dir = "/Users/johannesskaro/OneDrive - NTNU/summer-2023"
RESULTS_FOLDER = f"{onedrive_dir}/{sequence}"

def main():
    K = np.loadtxt(f"{RESULTS_FOLDER}/left/K_matrix.txt")
    R = np.loadtxt(f"{RESULTS_FOLDER}/left/R_matrix.txt")
    T = np.loadtxt(f"{RESULTS_FOLDER}/left/T_matrix.txt")

    plt.ion()

    left_images_filenames = list(filter(lambda fn: fn.split(".")[-1]=="png", os.listdir(f"{RESULTS_FOLDER}/left")))
    timestamps = list(map(lambda fn: fn.split(".")[0], left_images_filenames))
    timestamps = sorted(timestamps)
    for ti in range(0, len(timestamps)):
        timestamp = timestamps[ti]
        left = cv2.imread(f"{RESULTS_FOLDER}/left/{timestamp}.png")
        right = cv2.imread(f"{RESULTS_FOLDER}/right/{timestamp}.png")
        disp = np.array(cv2.imread(f"{RESULTS_FOLDER}/disp_zed/{timestamp}.png", cv2.IMREAD_ANYDEPTH) / 256.0, dtype=np.float32)
    
        plt.imshow(disp, cmap="turbo")
        plt.colorbar()
        plt.show()
        plt.pause(0.1)
        cv2.imshow("Left image", left)
        cv2.waitKey(100)
        plt.clf()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()