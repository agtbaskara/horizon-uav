import cv2
import math
import os
import sys

filename = sys.argv[1]
output_path = os.path.join("images")
interval = 10

file_path = os.path.join(filename)
cap = cv2.VideoCapture(file_path)
frameRate = cap.get(5)

while cap.isOpened():
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % (math.floor(frameRate) * interval) == 0:
        seek_time = str(int(frameId / (math.floor(frameRate))))
        output_filename = os.path.splitext(filename)[0] + "_" + \
            seek_time + ".jpg"
        output_file_path= os.path.join(output_path, output_filename)
        
        print(output_file_path)
        cv2.imwrite(output_file_path, frame)

cap.release()
