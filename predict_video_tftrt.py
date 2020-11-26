import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import tensorflow_addons as tfa
import cv2
import numpy as np
import os
import math
import time
import skvideo.io

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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

def draw_horizon_line(img, m, c, scale):
    """Draw horizon line on image"""
    image_height = img.shape[0]
    image_width = img.shape[1]

    c = scale*c

    pt1 = (0, int(m*0+c))
    pt2 = (image_width, int(m*image_width+c))

    cv2.line(img, pt1, pt2, (125, 0, 255), 2)

    return img

# Parameter
image_size = (224, 224)
model_path = os.path.join("model/model_unet_saved_model_TFTRT_FP32")
video_path = os.path.join("raw_dataset", "videos", "a3-converted.mp4-00.08.04.849-00.09.06.443.mp4")

# Load model
model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])

# Video Writer
outputfile = "output_tftrt.mp4"
writer = skvideo.io.FFmpegWriter(outputfile)

# Load Video
cap = cv2.VideoCapture(video_path)
while(cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    if ret:
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        frame = frame[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
        frame_ori = frame.copy()
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        frame = np.expand_dims(frame, axis=0)
        frame = tf.constant(frame)

        # Predict mask
        infer = model.signatures['serving_default']
        labeling = infer(frame)
        pred = labeling['activation'].numpy()

        # Process mask
        mask = pred.squeeze()
        mask = np.stack((mask,)*3, axis=-1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_land = mask[:, :, 0]
        mask_sky = mask[:, :, 1]
        
        # Post Process
        mask_land = cv2.cvtColor(mask_land, cv2.COLOR_BGR2GRAY)
        mask_sky = cv2.cvtColor(mask_sky, cv2.COLOR_BGR2GRAY)

        border = get_border(mask_land, mask_sky)
        m, c = get_horizon_line(border)

        resized_image_height = frame.shape[0]
        resized_image_width = frame.shape[1]
        roll, pitch = get_roll_pitch(m, c, resized_image_height, resized_image_width)
        
        #frame_ori = cv2.resize(frame_ori, (480, 480))
        scale = image_height/image_size[0]
        frame_ori = draw_horizon_line(frame_ori, m, c, scale)

        fps = 1.0/(time.time()-start_time)

        text_roll = "roll:" + str(round(roll, 2)) + " degree"
        text_pitch = "pitch:" + str(round(pitch, 2)) + " %"
        text_fps = "fps:" + str(round(fps, 2))

        cv2.putText(frame_ori, text_roll, (5, 15), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_pitch, (5, 35), 0, 0.5, (125, 0, 255), 2)
        cv2.putText(frame_ori, text_fps, (5, 55), 0, 0.5, (125, 0, 255), 2)

        writer.writeFrame(frame_ori[:,:,::-1])

    else:
        break

print("Video Ended")
cap.release()
writer.close()
