import numpy as np
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
import os

class FastSAMSeg:
    """
    A class to handle FastSAM segmentation tasks.
    """

    def __init__(self, model_path: str ='./weights/FastSAM-x.pt'):
        """
        Initialize the FastSAMSeg class.

        Parameters:
        - model_path (str): Path to the pretrained FastSAM model.
        """
        try:
            self.model = FastSAM(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading FastSAM model from {model_path}. Reason: {e}")
        
    def _segment_img(self, img: np.array, device: str = 'mps') -> np.array:
        """
        Internal method to perform segmentation on the provided image.

        Parameters:
        - img (np.array): Input image for segmentation.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Segmentation results.
        """
        retina_masks = True
        verbose = False
        results = self.model(img, device=device, retina_masks=retina_masks, verbose=verbose)
        return results

    def get_mask_at_points(self, img: np.array, points: np.array, pointlabel: np.array, device: str = 'mps') -> np.array:
        """
        Obtain masks for specific points on the image.

        Parameters:
        - img (np.array): Input image.
        - points (np.array): Array of points.
        - pointlabel (np.array): Corresponding labels for points.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        mask = np.zeros((img.shape[0], img.shape[1]))
        point_results = self.model(img, points=points, labels=pointlabel, device=device, retina_masks=True, verbose=False)
        ann = point_results[0].cpu()
        if len(ann)>0:
            mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_mask_at_bbox(self, img: np.array, bbox: np.array, device: str = 'mps') -> np.array:
        """
        Obtain masks for the bounding box on the image.

        Parameters:
        - img (np.array): Input image.
        - bbox (np.array): Bounding box.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Mask result.
        """
        box_results = self.model(img, bboxes=bbox, device=device, retina_masks=True)
        ann = box_results[0].cpu()
        mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

        return mask
    
    def get_all_masks(self, img: np.array, device: str = 'cuda') -> np.array:
        """
        Obtain all masks for the input image.

        Parameters:
        - img (np.array): Input image.
        - device (str): Device to run the model on, e.g., 'cuda'.

        Returns:
        - np.array: Masks result.
        """

        results = self._segment_img(img, device=device)
        if results[0].masks is not None:
            masks = np.array(results[0].masks.data.cpu())
        else: 
            masks = np.array([])
        return masks
    
    