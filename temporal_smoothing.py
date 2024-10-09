import numpy as np
from collections import deque

class TemporalSmoothing:

    N = 3
    past_N_masks = deque(maxlen=N)

    def __init__(self) -> None:
        pass

    def get_smoothed_water_mask(self, water_mask: np.array) -> np.array:
        """
        Get the smoothed water mask.

        Parameters:
        - water_mask (np.array): Water mask.

        Returns:
        - np.array: Smoothed water mask.
        """

        np
        mask_sum = np.sum(self.past_N_masks, axis=0)

        smoothed_water_mask = np.zeros_like(water_mask)
        threshold = self.N // 2
        thresholded_mask = (mask_sum > threshold).astype(int)

        smoothed_water_mask = np.logical_or(water_mask, thresholded_mask).astype(int)

        self.past_N_masks.append(water_mask)

        return smoothed_water_mask