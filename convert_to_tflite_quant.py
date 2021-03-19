import cv2
import tensorflow as tf
import numpy as np
import os

# Enable GPU Memory Growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_model_path = os.path.join("model/model-unet-modified.h5")
output_model_path = os.path.join("model/model-unet-modified-quant.tflite")

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth=tf.keras.backend.epsilon()
    dice=0
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
        dice += (intersection + smooth) / (union + smooth)
    return -2./num_classes * dice

# Create Representative Dataset
dataset_path = "dataset/images"
image_size = (128, 128)
file_list = [os.path.splitext(filename)[0] for filename in os.listdir(dataset_path)]
def representative_dataset():
    image_batch = []
    for filename in file_list:
        # Load Image
        image = cv2.imread(os.path.join("dataset", "images", \
            filename + ".jpg"))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, image_size)

        # Normalize
        image = cv2.normalize(image, None, 0, 1, \
            cv2.NORM_MINMAX, cv2.CV_32F)

        # Change Data Type
        image = image.astype(np.float32)

        # Append to batch
        image_batch.append(image)

    # Convert batch as array
    image_batch = np.array(image_batch)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_batch).batch(1)
    for i in image_dataset.take(1):
        print(i)
        yield [i]

# Load model
model = tf.keras.models.load_model(input_model_path, custom_objects={'dice_loss': dice_loss, 'MaxMeanIoU': MaxMeanIoU})

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save the model.
with open(output_model_path, 'wb') as f:
    f.write(tflite_model)
