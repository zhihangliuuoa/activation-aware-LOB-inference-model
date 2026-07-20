import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import subprocess
import re

print("="*80)
print("JETSON TENSORRT MEMORY USAGE MEASUREMENT")
print("="*80)

# Function to get GPU memory usage on Jetson
def get_jetson_gpu_memory():
    """
    Get GPU memory usage on Jetson using tegrastats
    Returns: (used_mb, total_mb)
    """
    try:
        # Method 1: Try using tegrastats (Jetson-specific)
        result = subprocess.run(['tegrastats', '--interval', '100'], 
                              capture_output=True, text=True, timeout=0.5)
    except:
        pass
    
    try:
        # Method 2: Read from /proc/meminfo or nvidia-smi equivalent
        # This is a simplified version - actual implementation depends on Jetson tools
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        # Parse memory info (this is approximate)
        print("Note: Memory measurement on Jetson - using approximate method")
        return None, None
    except:
        return None, None

def get_cuda_memory_info():
    """Get CUDA memory info using pycuda"""
    try:
        free, total = cuda.mem_get_info()
        used = total - free
        return used / (1024*1024), total / (1024*1024)  # Convert to MB
    except:
        return None, None

print("\nJetson GPU Memory Info:")
used, total = get_cuda_memory_info()
if used and total:
    print(f"  Total GPU Memory: {total:.0f} MB")
    print(f"  Used (before loading): {used:.0f} MB")
    print(f"  Free: {total - used:.0f} MB")

# ============================================================================
# Load TensorRT Engine
# ============================================================================
print("\n" + "="*80)
print("Loading TensorRT Engine")
print("="*80)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

mem_before_load = get_cuda_memory_info()

with open("cflob_fp16.engine", "rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

mem_after_load = get_cuda_memory_info()

if mem_before_load[0] and mem_after_load[0]:
    engine_memory = mem_after_load[0] - mem_before_load[0]
    print(f"Memory before engine load: {mem_before_load[0]:.2f} MB")
    print(f"Memory after engine load:  {mem_after_load[0]:.2f} MB")
    print(f"Engine size:               {engine_memory:.2f} MB")
else:
    print("Could not measure engine memory precisely")
    engine_memory = None

# ============================================================================
# Create Execution Context and Allocate Buffers
# ============================================================================
print("\n" + "="*80)
print("Creating Execution Context and Buffers")
print("="*80)

context = engine.create_execution_context()

input_shape = (1, 1, 10, 40)
output_shape = (1, 3)

# Calculate buffer sizes
input_size_bytes = int(np.prod(input_shape) * 2)  # FP16 = 2 bytes
output_size_bytes = int(np.prod(output_shape) * 2)

print(f"Input buffer size:  {input_size_bytes / 1024:.2f} KB")
print(f"Output buffer size: {output_size_bytes / 1024:.2f} KB")
print(f"Total I/O buffers:  {(input_size_bytes + output_size_bytes) / 1024:.2f} KB")

mem_before_buffers = get_cuda_memory_info()

# Allocate GPU buffers
d_input = cuda.mem_alloc(input_size_bytes)
d_output = cuda.mem_alloc(output_size_bytes)

mem_after_buffers = get_cuda_memory_info()

if mem_before_buffers[0] and mem_after_buffers[0]:
    buffer_memory = mem_after_buffers[0] - mem_before_buffers[0]
    print(f"\nMemory before buffer allocation: {mem_before_buffers[0]:.2f} MB")
    print(f"Memory after buffer allocation:  {mem_after_buffers[0]:.2f} MB")
    print(f"Buffer memory:                   {buffer_memory:.2f} MB")
else:
    buffer_memory = None

# ============================================================================
# Measure Peak Memory During Inference
# ============================================================================
print("\n" + "="*80)
print("Memory Usage During Inference")
print("="*80)

h_input = np.random.randn(*input_shape).astype(np.float16)
h_output = np.empty(output_shape, dtype=np.float16)

stream = cuda.Stream()

# Get tensor names
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

# Warmup
for _ in range(50):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

# Measure memory during inference
mem_before_inference = get_cuda_memory_info()

for _ in range(10):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

mem_after_inference = get_cuda_memory_info()

if mem_before_inference[0]:
    print(f"Memory during inference: {mem_before_inference[0]:.2f} MB")
    print(f"Memory after inference:  {mem_after_inference[0]:.2f} MB")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("JETSON TENSORRT MEMORY SUMMARY")
print("="*80)

total_used = get_cuda_memory_info()[0]
total_available = get_cuda_memory_info()[1]

print(f"\n{'Component':<30} {'Memory (MB)':<15}")
print("-" * 45)
if engine_memory:
    print(f"{'TensorRT Engine':<30} {engine_memory:.2f}")
if buffer_memory:
    print(f"{'I/O Buffers':<30} {buffer_memory:.2f}")
if total_used:
    print(f"{'Total Used':<30} {total_used:.2f}")
    print(f"{'Total Available':<30} {total_available:.2f}")
    print(f"{'Utilization':<30} {total_used/total_available*100:.1f}%")
print("-" * 45)

# ============================================================================
# Detailed Engine Information
# ============================================================================
print("\n" + "="*80)
print("TENSORRT ENGINE DETAILS")
print("="*80)

print(f"\nEngine Information:")
print(f"  Number of layers: {engine.num_layers}")
print(f"  Maximum batch size: {engine.max_batch_size}")
print(f"  Device memory size: {engine.device_memory_size / (1024*1024):.2f} MB")
print(f"  Number of bindings: {len([engine.get_tensor_name(i) for i in range(engine.num_io_tensors)])}")

print(f"\nTensor Bindings:")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"  [{i}] {name}:")
    print(f"      Shape: {shape}")
    print(f"      Dtype: {dtype}")
    print(f"      Mode:  {mode}")