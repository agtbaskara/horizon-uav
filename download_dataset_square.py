import os
import requests
import json
import cv2
import numpy as np
from pathlib import Path
from time import time as timer
from multiprocessing.pool import ThreadPool

def fetch_url(entry):
    """Download image and save it base on given url and path"""
    save_path, uri = entry
    if not os.path.exists(save_path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in r:
                    file.write(chunk)
    return save_path

# JSON file exported from labelbox
export_json_path = "dataset_export.json"

# Output path
images_output_path = os.path.join("dataset", "images")
masks_output_path = os.path.join("dataset", "masks")

# Make path if not exist
Path(images_output_path).mkdir(parents=True, exist_ok=True)
Path(masks_output_path).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "land")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(masks_output_path, "sky")).mkdir(parents=True, exist_ok=True)

# Open JSON
with open(export_json_path) as f:
    data = json.load(f)

start = timer()

counter = 0
dataset_size = len(data)
for i in range(0, dataset_size):
    filename = os.path.splitext(data[i]["External ID"])[0]

    if len(data[i]["Label"]) > 0: # Check if data labeled
        image_url = data[i]["Labeled Data"]
        image_path = os.path.join(images_output_path, filename + ".jpg")

        # Segmentation Label
        segmentation_label_size = len(data[i]["Label"]["objects"])

        if segmentation_label_size == 1: # Check if segmentation label exist
            print("Downloading", filename)
            fetch_url((image_path, image_url))
            counter += 1

            # Segmentation Label
            for j in range(0, segmentation_label_size):
                segmentation_label_value = (data[i]["Label"]["objects"][j]["value"])
                segmentation_label_url = (data[i]["Label"]["objects"][j]["instanceURI"])

                label_path = os.path.join(masks_output_path, segmentation_label_value, filename + ".png")
                fetch_url((label_path, segmentation_label_url))
                counter += 1
            
            # Square Crop Image
            image = cv2.imread(os.path.join(images_output_path, filename + ".jpg"))
            image_height = image.shape[0]
            image_width = image.shape[1]
            image = image[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
            cv2.imwrite(os.path.join(images_output_path, filename + ".jpg"), image)

            # Square Crop Mask Land
            image = cv2.imread(os.path.join(masks_output_path, "land", filename + ".png"))
            image_height = image.shape[0]
            image_width = image.shape[1]
            image = image[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
            cv2.imwrite(os.path.join(masks_output_path, "land", filename + ".png"), image)
            
            # Generate Sky Label
            mask_land = cv2.imread(os.path.join(masks_output_path, "land", filename + ".png"))
            mask_sky = cv2.bitwise_not(mask_land)

            cv2.imwrite(os.path.join(masks_output_path, "sky", filename + ".png"), mask_sky)

    else:
        print("Missing label", filename, "skippping")

print("Data downloaded: ", counter)
print("Elapsed Time:", timer() - start, "Seconds")
