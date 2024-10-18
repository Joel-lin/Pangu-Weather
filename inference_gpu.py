import os
import numpy as np
import onnx
import torch
#import onnxruntime as ort
import matplotlib.pyplot as plt
from onnx2torch import convert

# The directory of your input and output data
input_data_dir = 'input_data'
output_data_dir = 'output_data'
model = onnx.load('pangu_weather_24.onnx')
model_24 = convert(model)

model_24.eval()

import intel_extension_for_pytorch as ipex
model_24 = model_24.to('xpu')
#data = data.to('xpu')
model_24 = ipex.optimize(model_24)

# Set the behavier of onnxruntime
#options = ort.SessionOptions()
#options.enable_cpu_mem_arena=False
#options.enable_mem_pattern = False
#options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
#options.intra_op_num_threads = 1

# Set the behavier of cuda provider
#cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

# Initialize onnxruntime session for Pangu-Weather Models
#ort_session_24 = ort.InferenceSession('pangu_weather_24.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

# Load the upper-air numpy arrays
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays

input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
input_tensor1 = torch.from_numpy(input_upper)
input_tensor2 = torch.from_numpy(input_surface)
input_tensor1 = input_tensor1.to('xpu')
input_tensor2 = input_tensor2.to('xpu')

with torch.no_grad():
    starttime = time()
    output_tensor = model_24(input_tensor1, input_tensor2)
    endtime = time()
    inference_time = endtime - starttime
# Run the inference session
#output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
# Save the results
#np.save(os.path.join(output_data_dir, 'output_upper'), output)
#np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)

#np.save(os.path.join(output_data_dir, 'output_upper'), output_tensor[0])
#np.save(os.path.join(output_data_dir, 'output_surface'), output_tensor[1])
print("inference time: %.3f", inference_time)

#plt.imshow(output_tensor[1][3].to('cpu') - 273.15)
#plt.colorbar()
#plt.savefig("./panguval2018-09-27.png", format=png)
