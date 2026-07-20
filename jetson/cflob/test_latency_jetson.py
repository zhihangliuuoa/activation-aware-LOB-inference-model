import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("cflob_fp16.engine", "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_shape = (1, 1, 10, 40)
output_shape = (1, 3)

# Convert to int() to avoid numpy.int64 type mismatch
d_input = cuda.mem_alloc(int(np.prod(input_shape) * 2))
d_output = cuda.mem_alloc(int(np.prod(output_shape) * 2))

h_input = np.random.randn(*input_shape).astype(np.float16)
h_output = np.empty(output_shape, dtype=np.float16)

stream = cuda.Stream()

# Get input/output tensor names
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

# Set tensor addresses for the context
context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

# warmup
for _ in range(20):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

t0 = time.time()
for _ in range(200):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
t1 = time.time()

print("Latency (ms):", (t1 - t0) * 1000 / 200)