import numpy as np
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import cv2
from stixels import Stixels
import bev

# Load model
model_path = "weights/FastSAM-s.pt"
model = FastSAM(model_path)
device = "mps"

# Load image
img_path = "data/00022351L.jpg"
img = cv2.imread(img_path)
imgsz = img.shape
width, height = imgsz[1], imgsz[0]

# Run inference on an image
#everything_results = model(img_path, device=device, retina_masks=True, verbose=False)
point_results = model(img_path, points=[(width-1)/2, height-200],labels=[1], texts="water", device=device, retina_masks=True) #point prompt bottom center pixel
ann = point_results[0].cpu()
mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))


#predictor = FastSAMPredictor(device=device, retina_masks=True, verbose=False)
#results = predictor(img)
#ann = predictor.prompt(results, points=[(width-1)/2, height-200], pointlabel="water")

mask = np.array(ann[0].masks.data[0]) if ann[0].masks else np.zeros((img.shape[0], img.shape[1]))

stixel = Stixels()
stixel_mask = stixel.get_stixels(mask)



# Display results
cv2.imshow("mask", mask)
cv2.waitKey(0)

cv2.imshow("mask", stixel_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

for result in point_results:
    result.show()