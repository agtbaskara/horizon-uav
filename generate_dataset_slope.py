import glob
import pandas as pd
import os
import cv2
import numpy as np
import math

def get_border(mask_land, mask_sky):
    """Get horizon border image from land and sky mask"""
    # Convert Colorspace to Grayscale
    mask_land = mask_land[:,:]
    mask_sky = mask_sky[:,:]
    # Get Horizon Border Using Dilation and Bitwise AND
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    land_dilated = cv2.dilate(mask_land, kernel)
    sky_dilated = cv2.dilate(mask_sky, kernel)
    border = cv2.bitwise_and(land_dilated, sky_dilated)

    return border

def get_horizon_line(border):
    """Get horizon line equation from border image"""
    # Get border data in x,y format
    y = np.argmax(border, axis=0)
    x = np.arange(len(y))
    border_data = np.vstack([x, y]).T

    # Remove 0 from border data
    border_data = border_data[border_data[:, -1] != 0]

    # Linear Regression using border data
    # y = m*x+c
    x = border_data[:,0]
    y = border_data[:,1]

    X = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return m, c

def get_roll_pitch(m, c, image_height, image_width):
    """Get roll and pitch from horizon line equation"""
    # Convert slope (m) to roll degrees
    roll = math.degrees(math.atan(m))

    # Get pitch
    pitch = ((m*(image_width/2)+c)-(image_width/2))/(image_width/2)*100
    
    return roll, pitch

def draw_horizon_line(img, m, c):
    """Draw horizon line on image"""
    image_height = img.shape[0]
    image_width = img.shape[1]

    pt1 = (0, int(m*0+c))
    pt2 = (image_width, int(m*image_width+c))

    cv2.line(img, pt1, pt2, (125,0,255), 3)

    return img

file_list = glob.glob("dataset\images\*")

# Create dataframe for slope and offset
column_names = ["filename", "slope", "offset"]
df = pd.DataFrame(columns = column_names)

for filename in file_list:
    filename_images = os.path.split(filename)[1]
    filename = os.path.splitext(os.path.split(filename)[1])[0]

    images_path = os.path.join("dataset", "images", filename+".jpg")
    sky_path = os.path.join("dataset", "masks", "sky", filename+".png")
    land_path = os.path.join("dataset", "masks", "land", filename+".png")

    frame_ori = cv2.imread(images_path)
    mask_sky = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
    mask_land = cv2.imread(land_path, cv2.IMREAD_GRAYSCALE)

    image_height = frame_ori.shape[0]
    image_width = frame_ori.shape[1]

    border = get_border(mask_land, mask_sky)
    m, c = get_horizon_line(border)
    c = c/image_height
    #roll, pitch = get_roll_pitch(m, c, image_height, image_width)

    new_row = {"filename": filename_images, "slope": m, "offset": c}
    df = df.append(new_row, ignore_index=True)

# Save labels to csv file
df.to_csv("label.csv", index = False)

