import matplotlib.pyplot as plt
import numpy as np
import cv2



def plot_disparity_column(disparity_img, stixel_mask, column):

    
    disparity_values = disparity_img[:, column]
    stixel_mask_column = stixel_mask[:, column]
    stixel_indices = [i for i, val in enumerate(stixel_mask_column) if val == 1]
    stixel_disparity_values = [disparity_values[i] for i in stixel_indices]


    plt.figure(figsize=(10, 5))
    plt.plot(disparity_values, range(len(disparity_values)), color='blue', linewidth=4) 
    plt.plot(stixel_disparity_values, stixel_indices, color='red', linewidth=4) 
    plt.gca().invert_yaxis()
    plt.ylabel('Height [px]')
    plt.xlabel('Disparity [px]')
    plt.legend()
    plt.show(block=True)



def plot_stixel_img_without_column(img, stixel_mask, column, width=5):
    red_width = width + 3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_highlighted = img.copy()

    # Ensure the image has three color channels (RGB)
    if len(img_highlighted.shape) == 2:  # If the image is grayscale
        img_highlighted = np.stack([img_highlighted] * 3, axis=-1)

    # Define the column range for the width
    col_start = max(0, column - width // 2)
    col_end = min(img_highlighted.shape[1], column + width // 2 + 1)

    # Set the specified column range to blue (set the RGB values)
    img_highlighted[:, col_start:col_end] = [0, 0, 255]  # Blue color for the entire column range

        # Define the column range for the red overlay
    red_col_start = max(0, col_start - red_width // 2)
    red_col_end = min(img_highlighted.shape[1], col_end + red_width // 2)

    # Iterate through the range and set red pixels where the stixel mask is 1
    for col in range(red_col_start, red_col_end):
        stixel_mask_column = stixel_mask[:, col]
        red_indices = np.where(stixel_mask_column == 1)[0]
        img_highlighted[red_indices, col] = [255, 0, 0]  # Red color for specific pixels

    # Plot the modified image
    plt.figure(figsize=(10, 5))
    plt.imshow(img_highlighted)
    plt.ylabel('Height [px]')
    plt.xlabel('Width [px]')
    plt.legend()
    plt.show(block=True)