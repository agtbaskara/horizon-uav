from tensorflow.python.compiler.tensorrt import trt_convert as trt

print('Converting to TF-TRT FP32...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32, max_workspace_size_bytes=1000000000)

converter = trt.TrtGraphConverterV2(input_saved_model_dir='model/model_unet_saved_model', conversion_params=conversion_params)

converter.convert()
converter.save(output_saved_model_dir='model/model_unet_saved_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')
