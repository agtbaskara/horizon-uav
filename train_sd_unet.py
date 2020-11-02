import tensorflow as tf
import tensorflow_addons as tfa
import os
import datetime
import random
import numpy as np
import math
import cv2
import pandas as pd
import pathlib
import albumentations as albu
from sklearn.model_selection import train_test_split
from time import time as timer

# Training parameter
test_size = 0.2
random_seed = 1234

# Hyperparameter
epoch = 50
batch_size = 32
learning_rate = 0.0001
image_size = (128, 128)

# Augmentation
transformations = [albu.HorizontalFlip(p=0.5),
                   albu.VerticalFlip(p=0.5),
                   albu.ShiftScaleRotate(p=0.5),
                   albu.RandomBrightnessContrast(p=0.5),
                   albu.RandomGamma(p=0.5),
                   albu.ElasticTransform(p=0.5),
                   albu.GaussNoise(p=0.5),
                   albu.HueSaturationValue(p=0.5)
                   ]

aug = albu.Compose(transformations)

# Data Generator
class data_generator(tf.keras.utils.Sequence):
    def __init__(self, file_list, batch_size, image_size, \
        shuffle=True, augmentation=None):

        self.file_list = file_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.aug = augmentation
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*\
            self.batch_size]

        batch = [self.file_list[k] for k in indexes]

        # Create batch list
        batch_x = []
        batch_y = []

        for filename in batch:
            # Load Image
            image = cv2.imread(os.path.join("dataset", "images", \
                filename + ".jpg"))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Load mask
            mask_land = cv2.imread(os.path.join("dataset", \
                "masks", "land", filename + ".png"), \
                cv2.IMREAD_GRAYSCALE)

            mask_sky = cv2.imread(os.path.join("dataset", \
                "masks", "sky", filename + ".png"), \
                cv2.IMREAD_GRAYSCALE)

            mask = np.dstack((mask_land, mask_sky))

            # Resize
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, \
                interpolation = cv2.INTER_NEAREST)

            # Augmentation
            if self.aug is not None:
                augmented = self.aug(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Normalize
            image = cv2.normalize(image, None, 0, 1, \
                cv2.NORM_MINMAX, cv2.CV_32F)

            mask = cv2.normalize(mask, None, 0, 1, \
                cv2.NORM_MINMAX, cv2.CV_32F)

            # Load to batch
            batch_x.append(image)
            batch_y.append(mask)

        # Convert batch as array
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        return batch_x, batch_y

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth = tf.keras.backend.epsilon()
    dice = 0
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + \
            tf.keras.backend.sum(y_pred_f)
        dice -= (2. * intersection + smooth) / (union + smooth)
    return dice/num_classes

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

# Create model
def create_model():
    # Input
    input_shape = (image_size[0], image_size[1], 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    encoder_layers = []

    # Block 1 Encoder
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x)
    encoder_layers.append(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Block 2 Encoder
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    encoder_layers.append(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Block 3 Encoder
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    encoder_layers.append(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Block 4 Encoder
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    encoder_layers.append(x)
    x = tf.keras.layers.MaxPool2D()(x)

    # Block 5 Encoder
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    print(len(encoder_layers))

    # Block 1 Decoder
    x = tf.keras.layers.UpSampling2D(size=2)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, encoder_layers[3]])

    # Block 2 Decoder
    x = tf.keras.layers.UpSampling2D(size=2)(x)
    x = tf.keras.layers.Concatenate()([x, encoder_layers[2]])
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same')(x)

    # Block 3 Decoder
    x = tf.keras.layers.UpSampling2D(size=2)(x)
    x = tf.keras.layers.Concatenate()([x, encoder_layers[1]])
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=1, padding='same')(x)

    # Block 4 Decoder
    x = tf.keras.layers.UpSampling2D(size=2)(x)
    x = tf.keras.layers.Concatenate()([x, encoder_layers[0]])
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=1, padding='same')(x)

    # Block 5 Decoder
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=1, padding='same')(x)
    x = tfa.layers.GroupNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Conv2D(2, kernel_size=1, padding='same')(x)
    outputs = tf.keras.layers.Activation('softmax')(x)
    
    # Create Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create Loss Function
    loss = dice_loss

    # Create Model
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = opt, loss = loss, metrics=["accuracy", MaxMeanIoU(num_classes=2)])
    
    return model

# Create Callback
def create_callback():
    # Tensorboard Callbacks
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # Checkpoint Callbacks
    pathlib.Path("checkpoint").mkdir(parents=True, exist_ok=True)

    best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join("checkpoint", "best" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"),
                                                             monitor='max_mean_io_u', verbose=1, save_best_only=True, mode='max')

    # Predict Image Callbacks
    file_writer = tf.summary.create_file_writer(os.path.join(logdir, "predict_output"))
    def predict_epoch(epoch, logs):
        # Load image
        filename = np.random.choice(test_list)
        image = cv2.imread(os.path.join("dataset", "images", filename + ".jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # Predict mask
        pred = model.predict(np.expand_dims(image, 0))

        # Process mask
        mask = pred.squeeze()
        mask = np.stack((mask,)*3, axis=-1)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        class_land = np.concatenate([image, mask[:, :, 0], image * mask[:, :, 0]], axis = 1)
        class_sky = np.concatenate([image, mask[:, :, 1], image * mask[:, :, 1]], axis = 1)

        # Log the image as an image summary.
        with file_writer.as_default():
            tf.summary.image("class_land", np.reshape(class_land, (1, image_size[0], image_size[1]*3, 3)), step=epoch)
            tf.summary.image("class_sky", np.reshape(class_sky, (1, image_size[0], image_size[1]*3, 3)), step=epoch)

    # Define per-epoch callback.
    predict_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=predict_epoch)

    return [tensorboard_callback, best_checkpoint_callback, predict_callback]

# Training

# Load Data
dataset_path = "dataset/images"
file_list = [os.path.splitext(filename)[0] for filename in os.listdir(dataset_path)]

# Data Split
train_list, test_list = train_test_split(file_list, shuffle=True, \
    test_size=test_size, random_state=random_seed)

start = timer()

# Load data
train_generator = data_generator(train_list, \
    batch_size=batch_size, image_size=image_size, \
    augmentation=aug)

val_generator = data_generator(test_list, \
    batch_size=batch_size, image_size=image_size)

# Create model
model = create_model()

# Train model
model.fit(train_generator, epochs=epoch, \
    validation_data=val_generator,\
    callbacks=create_callback(), max_queue_size=5)

# Evaluate Model
result = model.evaluate(val_generator)

loss = result[0]
accuracy = result[1]
mean_io_u = result[2]

# Get evaluation metric
print("Run Date:", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
print("Elapsed Time:", timer() - start, "Seconds")
print("Training parameter")
print("test_size:", test_size)
print("random_seed:", random_seed)
print()
print("Hyperparameter:")
print("epoch:", epoch)
print("batch_size:", batch_size)
print("learning_rate:", learning_rate)
print()
print("Result:")
print("loss:", loss)
print("accuracy:", accuracy)
print("mean_io_u:", mean_io_u)
