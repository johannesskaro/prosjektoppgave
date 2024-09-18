import numpy as np
import cv2
import matplotlib.pyplot as plt
# from utilities import calculate_3d_points, calculate_3d_point, calculate_3d_points_from_mask

def calculate_colors_from_mask(mask, rgb_image):
    Y, X = np.where(mask > 0)
    colors = rgb_image[Y, X]
    return colors


def calculate_bev_image(points_3d, y_threshold=4, colors=None, scale_factor=1, image_size=(500, 500), line_interval=10, max_depth=40):
    """
    Plots the Bird's Eye View of 3D points using OpenCV and draws thin horizontal lines for every 10m.
    Args:
        points_3d (numpy.ndarray): An array of 3D points, shape (N, 3)
        y_threshold (float): Threshold for filtering points based on their Y-coordinate
        colors (numpy.ndarray): An array of colors for each point
        scale_factor (float): Factor to scale the points for better visualization
        image_size (tuple): Size of the output image (width, height)
        line_interval (int): Interval in meters for drawing horizontal lines
    """
    if points_3d.shape[1] != 3:
        raise ValueError("Input points must be 3D (N, 3) shape.")

    # Create a blank image
    bev_image = 255 * np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    min_y = -y_threshold
    max_y = y_threshold
    mask = ((points_3d[:, 1] - 2) >= min_y) & ((points_3d[:, 1] - 2) <= max_y)
    filtered_points = points_3d[mask]

    # If colors are provided, they need to be filtered as well
    filtered_colors = colors[mask] if colors is not None else None

    # Extracting X and Z for BEV (Bird's Eye View)
    X_unorm, Z_unorm = filtered_points[:, 0], filtered_points[:, 2]
    
    Z_mask = Z_unorm<=max_depth
    Z_unorm = Z_unorm[Z_mask]
    X_unorm = X_unorm[Z_mask]

    if not np.sum(X_unorm) or not np.sum(Z_unorm):
        return bev_image

    if X_unorm.shape[0] == 0 or Z_unorm.shape[0] == 0:
        return bev_image

    X = (X_unorm - np.min(X_unorm)) / (np.max(X_unorm) - np.min(X_unorm)) * image_size[0]
    Z = (Z_unorm - np.min(Z_unorm)) / (np.max(Z_unorm) - np.min(Z_unorm)) * image_size[1]

    filtered_colors = filtered_colors[Z_mask] if colors is not None else None
    
    camera_loc_x = int((0 - np.min(X_unorm)) / (np.max(X_unorm) - np.min(X_unorm)) * image_size[0])
    camera_loc_y = int((0 - np.min(Z_unorm)) / (np.max(Z_unorm) - np.min(Z_unorm)) * image_size[0])
    
    #print((camera_loc_x,camera_loc_y))
    y_shift = 0 - camera_loc_y
    cv2.circle(bev_image, (camera_loc_x,camera_loc_y + y_shift), radius=3*int(scale_factor), color=(0,191,255), thickness=-1)

    # Draw points
    for i, xz in enumerate(zip(X, Z)):
        (x, z) = xz
        cv2.circle(bev_image, (int(x), int(z) + y_shift), radius=int(scale_factor), color=filtered_colors[i].tolist(), thickness=-1)

    for depth in range(line_interval, max_depth + 1, line_interval):
        normalized_depth = (depth - np.min(Z_unorm)) / (np.max(Z_unorm) - np.min(Z_unorm)) * image_size[1]
        cv2.line(bev_image, (0, int(normalized_depth)+y_shift), (image_size[0], int(normalized_depth)+y_shift), (0, 0, 0), 1)

    # Flip the image vertically for correct orientation
    bev_image = cv2.flip(bev_image, 0)
    return bev_image

def update_bev_plot(fig, ax, points_3d, y_threshold = 4, colors=None, scale_factor=1):
    """
    Updates the Bird's Eye View plot with new 3D points.
    """
    ax.clear()  # Clear the previous plot

    min_y = -y_threshold
    max_y = y_threshold
    mask = ((points_3d[:, 1] - 2) >= min_y) & ((points_3d[:, 1] - 2) <= max_y)
    filtered_points = points_3d[mask]

    # If colors are provided, they need to be filtered as well
    filtered_colors = colors[mask] if colors is not None else None

    # Extracting X and Z for BEV (Bird's Eye View)
    X, Z = filtered_points[:, 0], filtered_points[:, 2]

    Z_mask = Z<=40
    Z = Z[Z_mask]
    X = X[Z_mask]
    filtered_colors = filtered_colors[Z_mask] if colors is not None else None

    if colors is not None:
        ax.scatter(X, Z, c=filtered_colors, s=scale_factor)  # Update with new points and colors
    else:
        ax.scatter(X, Z, s=scale_factor)  # Update with new points
    # ax.axis('off')
    # ax.set_title('Bird\'s Eye View')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    # ax.axis([0,axis[0],0,axis[1]])  # Ensures equal aspect ratio
    ax.axis('equal')


    plt.draw()
    plt.pause(0.01)  # Pause briefly to allow the plot to be updated