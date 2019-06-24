#!/usr/bin/env python3

import io
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import utils

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#runtime = trt.Runtime(TRT_LOGGER)

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


with open('yolov3_fp16.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
  engine = runtime.deserialize_cuda_engine(f.read())

#with open("yolov3_fp16.engine", 'rb') as f:
#    with runtime.deserialize_cuda_engine(f.read()) as engine:
#        inputs, outputs, bindings, stream = allocate_buffers(engine)
#        with engine.create_execution_context() as context:
#            img = cv2.imread("006850.png")
#            input = utils.image_preporcess(img.copy(), [416, 416])
#            input = np.transpose(i, (2,0,1))
#            np.copyto(inputs[0].host, input)
#            preds = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
