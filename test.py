import numpy as np
import tensorflow as tf
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/model-unet.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread("dataset/images/e12-converted.mp4-00.01.20.478-00.02.38.637_100.jpg")
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
input_data = np.array(img, dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)

# Test the model on random input data.
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

#mask = output_data.squeeze()
#mask = np.stack((mask,)*3, axis=-1)
#mask[mask >= 0.5] = 1
#mask[mask < 0.5] = 0

mask_land = output_data[0, :, :, 0]
mask_sky = output_data[0, :, :, 1]

cv2.imshow("Horizon", mask_sky)
cv2.waitKey(0)