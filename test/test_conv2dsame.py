import torch
import io
import onnxruntime
import onnx
from layers.convolutions.conv2d_utils import Conv2dSame
import numpy as np
from onnx2keras import onnx_to_keras
import tensorflow as tf

keras_output = "demo_keras.h5"
tflite_output = "tflite_model.tflite"
temp_f = "model.onnx"
im_size = 256

input_data = np.random.uniform(0, 1, (1, 3, im_size, im_size)).astype(np.float32)
model = Conv2dSame(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=[1, 1, 1, 1],
    dilation=1,
    groups=1,
    bias=True
)
model.eval()

input_variable = torch.FloatTensor(input_data)
input_names = ['test_in']

torch.onnx.export(model, input_variable, temp_f, verbose=False, input_names=input_names,
                  output_names=['test_out'], export_params=True)

onnx_model = onnx.load(temp_f)
k_model = onnx_to_keras(
    onnx_model, input_names, 
    change_ordering=True,
    padding_mode='same'
)
k_model.save(keras_output)


# convert to TfLite
model = tf.keras.models.load_model(keras_output)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_output, "wb") as f:
    f.write(tflite_model)

print("Keras -> TfLite is Done")


onnx_session = onnxruntime.InferenceSession(temp_f)
onnx_inputs = {onnx_session.get_inputs()[0].name: input_data}
onnx_outputs = onnx_session.run(None, onnx_inputs)

interpreter = tf.lite.Interpreter(model_path=str(tflite_output))
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(
    input_details[0]["index"], (1, im_size, im_size, 3), strict=True
)
interpreter.allocate_tensors()
interpreter.set_tensor(
    input_details[0]["index"], input_data.transpose(0, 2, 3, 1)
)
interpreter.invoke()

tflite_output_index = output_details[0]['index']
print(f"Output tflife index: {tflite_output_index}")
onnx_out = onnx_outputs[0]
tflite_out = interpreter.get_tensor(
    tflite_output_index).transpose(0, 3, 1, 2)
print(f"Shape onnx = {onnx_out.shape}\ttflite = {tflite_out.shape}")
mse = ((onnx_out - tflite_out) ** 2).mean(axis=None)
mae = np.abs(onnx_out - tflite_out).mean(axis=None)
print(
    f"MSE (Mean-Square-Error): {mse}\tMAE (Mean-Absolute-Error): {mae}"
)
print("===========================")
