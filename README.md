# horizon-uav

<p align="center">
<img src="https://user-images.githubusercontent.com/15165276/111776126-35b66900-88e4-11eb-8c49-14efe0e8094e.gif" width="50%" ></img>
</p>

Detect horizon using front-faced mounted camera on UAV using deep learning based computer vision algorithm. Tested to run in real-time using Raspberry Pi 4B + Google Coral Edge TPU + Raspberry Pi Camera v1.

## Requirment:
- Python 3
- Tensorflow
- OpenCV
- Numpy
- Tensorflow-lite (Optional)
- TensorRT (Optional)

## Hardware Requirment:
- Computer with decend RAM and CPU
- Nvidia GPU or Jetson Dev Board (Optional)
- Raspberry Pi 4B (Optional)
- Camera (Optional)
- Google Coral USB Accelerator (Optional)

## How to Use:
### Dataset:
- Use `download_dataset_original.py` to download dataset in original ratio
- Use `download_dataset_square.py` to download dataset in square ratio
- You can also download the dataset and sample video manually from here: https://1drv.ms/f/s!ArMhy7w0Pabls18kV9D7GVtonJmq
- Store dataset at `dataset/images`

Notes: Change `export_json_path` to the appropriate json file exported from LabelBox

### Training:
- Use `train_*.py` to train your desired architecture (`unet`, `sd-unet`, or `mobilenet-unet`)
- Change `dataset_path` to the appropriate path if needed
- You can modify the Hyperparameter and Augmentation if needed
- Refer `train_google_colab.ipynb` to train model using google colab

### Quantization:
- Use `convert_to_tflite.py` to convert your Tensorflow model to TFLite model
- Use `convert_to_tflite_quant.py` to convert your TFLite model to 8-Bit Quantized TFLite model
- Use `convert_to_tftrt.py` to convert your Tensorflow model to TensorRT model
- Change `input_model_path` and `output_model_path` to the appropriate path if needed

Notes: Not all architecture support quantization

### Inference:
- Train a model or download pre-trained model from here: https://1drv.ms/f/s!ArMhy7w0Pabls18kV9D7GVtonJmq
- Use `predict_image.py` to do inference on single image
- Use `predict_camera_*.py` to do inference using camera as input
- Use `predict_video_*.py` to do inference using video as input
- Change `model_path`, `image_path`, or `video_path` to the appropriate path if needed

Notes: `edgetpu` model require Google Coral, `tflite` model require Tensorflow-lite installed, `tftrt` require Nvidia GPU and TensorRT installed

### Additional:
- `test.py` is legacy program to test inference
- `post_process.py` is legacy program to test post inference process
- `save_model.py` is legacy program to save model as `SavedModel` format

## TODO:
- Add communication protocol to Flight Controller
- Change Linear Regression to Line Fitting algorithm (ex: RANSAC) for more robust detection
- Update dataset to add real attitude data captured from UAV
- Change architecture for direct attitude output without using post processing (Direct Image to Attitude Model)
    - Try to use classification (By seperating each degrees as seperate class)
    - Or just do a direct regression
- Just play with it! Add LSTM or something interesting (Don't forget it must run in real-time!)

## How it works:
The horizon detection system in the UAV aims to detect the horizon line to obtain UAV orientation data relative to the earth's horizon using AI and computer vision. This orientation data is expected to back up IMU sensors and help control systems in the UAV. This horizon detection system is run on a Single Board Computer connected to the Flight Controller. The Single Board Computer is also equipped with a camera and AI Accelerator, as shown below:
<p align="center">
    <img src="https://user-images.githubusercontent.com/15165276/111776145-3cdd7700-88e4-11eb-9b0a-9d2ad56a92ff.png" width="50%"></img>
</p>
There are 4 stages in the horizon detection system:

1. Semantic Segmentation
    <p align="center">
        <img src="https://user-images.githubusercontent.com/15165276/111776211-4e268380-88e4-11eb-877a-d18f79ce9ae7.png" width="50%"></img> 
    </p>
    In the first stage, semantic segmentation is carried out on the input image taken from the camera facing the front of the UAV. Semantic segmentation aims to detect the position of heaven and earth in the image. This process is carried out using a previously trained U-Net-based deep learning algorithm
    using a labeled dataset. The output of this stage is a binary image showing the position of the heavens and the earth.

2. Border Extraction
    <p align="center">
        <img src="https://user-images.githubusercontent.com/15165276/111776162-41a22b00-88e4-11eb-8783-a4d58dd19656.png" width="50%"></img>
    </p>
    In the second stage, the boundary line's extraction between heaven and earth is carried out using bitwise operations on the image. At this stage, morphology operations are also carried out on the image to reduce noise in the image obtained in the semantic segmentation process. The output from this stage is data from the horizon line that separates the sky and earth.

3. Linear Regression
    <p align="center">
        <img src="https://user-images.githubusercontent.com/15165276/111776139-39e28680-88e4-11eb-92ab-6d72ba992b93.png" width="50%"></img>
    </p>
    In the third stage, linear regression is carried out on the horizon line data to obtain the line equation in the form 𝑦 = 𝑚 ∗ 𝑥 + 𝑐

4. Roll and Pitch calculation
    In the last stage, the previously obtained horizon line equation is converted into roll and pitch angles using these formula:
    <p align="center">
        <img src="https://user-images.githubusercontent.com/15165276/111776201-4bc42980-88e4-11eb-91db-98502bd441ff.png" width="50%"></img>
    </p>
    <p align="center">
        <img src="https://user-images.githubusercontent.com/15165276/111783285-a3669300-88ec-11eb-888f-d5c705ad3709.jpg" width="50%"></img>
    </p>
    The obtained pitch and roll values ​​will then be sent to the flight controller for use to control the vehicle's stability.

## Reference:
- O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv:1505.04597 [cs], May 2015. Available: http://arxiv.org/abs/1505.04597.
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” arXiv:1801.04381 [cs], Mar. 2019. Available: http://arxiv.org/abs/1801.04381.
- P. K. Gadosey et al., “SD-UNet: Stripping down U-Net for Segmentation of Biomedical Images on Platforms with Low Computational Budgets,” Diagnostics, vol. 10, no. 2, p. 110, Feb. 2020, doi: 10.3390/diagnostics10020110.
- Tensorflow Documentation, https://www.tensorflow.org/api_docs/
- OpenCV Documentation, https://docs.opencv.org/master/
- Numpy Linear Algorithm Documentation, https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
- Google Coral Documentation, https://coral.ai/docs/
- Nvidia TensorRT Documentation, https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
