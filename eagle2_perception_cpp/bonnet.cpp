#include <stdlib.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
// Nvidia stuff
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "bonnet.hpp"

using namespace std;
using namespace nvinfer1;

namespace bonnet
{

Bonnet::Bonnet(const string& engineFile)
{
  string input_node="test_model/model/images/truediv";
  string output_node="softmax_1";
  num_classes = 20;
  d = 3;
  w = 512;
  h = 256;

  // read engine from file
  fstream file;
  file.open(engineFile, ios::binary | ios::in);
  if(!file.is_open())
  {
      cout<<"bonnet: failed to read engine file "<<engineFile<<endl;
      return;
  }
  file.seekg(0, ios::end);
  int length = file.tellg();
  file.seekg(0, ios::beg);
  unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();

  // initialize plugins, required for LeakyRelu
  if (!initLibNvInferPlugins(&logger, "")) return;

  // create engine
  runtime = createInferRuntime(logger);
  assert(runtime != nullptr);
  engine=runtime->deserializeCudaEngine(data.get(), length, nullptr);
  assert(engine != nullptr);

  // create an execution context
  context = engine->createExecutionContext();
  assert(context != nullptr);
  // Get the bindings for input and output
  inputIndex  = engine->getBindingIndex(input_node.c_str());
  inputDims   = engine->getBindingDimensions(inputIndex);
  outputIndex = engine->getBindingIndex(output_node.c_str());
  outputDims  = engine->getBindingDimensions(outputIndex);
  // sizes for cuda malloc
  size_in_pix = w * h;
  sizeof_in = d * size_in_pix * sizeof(float);
  sizeof_out = num_classes * size_in_pix * sizeof(int);
  // Allocate GPU memory for I/O
  cuda_buffers[inputIndex] = &input_gpu;
  cuda_buffers[outputIndex] = &output_gpu;
  cudaMalloc(&cuda_buffers[inputIndex], sizeof_in);
  cudaMalloc(&cuda_buffers[outputIndex], sizeof_out);
  // Use CUDA streams to manage the concurrency of copying and executing
  cudaStreamCreate(&cuda_stream);

  // we are good
  initialized=true;
}

Bonnet::~Bonnet()
{
  // Release the stream and the buffers
  cudaStreamSynchronize(cuda_stream);
  cudaStreamDestroy(cuda_stream);
  for(auto& item : cuda_buffers)
      cudaFree(item);
  if(!runtime)
      runtime->destroy();
  if(!context)
      context->destroy();
  if(!engine)
      engine->destroy();
};

void Bonnet::doInference(const cv::Mat& image, cv::Mat& output)
{
  // resize image with preserving aspect ratio
  float scale = min(float(w)/image.cols,float(h)/image.rows);
  auto scaleSize = cv::Size(image.cols*scale,image.rows*scale);
  cv::Mat resized;
  cv::resize(image,resized,scaleSize,0,0,cv::INTER_LINEAR);
  cv::Mat cropped(w, h, CV_8UC3, 127);
  cv::Rect rect((w-scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width, scaleSize.height);
  resized.copyTo(cropped(rect));

  // normalize
  cv::Mat input;
  cropped.convertTo(input, CV_32FC3);
  input = (input - 128.0f) / 128.0f;

  vector<cv::Mat> input_chw(d);
  cv::split(input, input_chw);
  // copy the B, G and, R data into a contiguous array
  vector<float> input_chw_data;
  for (unsigned int ch = 0; ch < d; ++ch) {
    if (input_chw[ch].isContinuous()) {
      input_chw_data.insert(input_chw_data.end(),
                                 (float*)input_chw[ch].datastart,
                                 (float*)input_chw[ch].dataend);
    } else {
      for (unsigned int y = 0; y < h; ++y) {
        input_chw_data.insert(input_chw_data.end(),
                                   input_chw[ch].ptr<float>(y),
                                   input_chw[ch].ptr<float>(y) + w);
      }
    }
  }

  // Copy Input Data to the GPU memory
  cudaMemcpyAsync(cuda_buffers[inputIndex], input_chw_data.data(),
                  sizeof_in, cudaMemcpyHostToDevice, cuda_stream);


  // Enqueue the op
  context->enqueue(1, cuda_buffers, cuda_stream, nullptr);

  // Copy Output Data to the CPU memory
  vector<float> output_chw(size_in_pix * num_classes);
  cudaMemcpyAsync(output_chw.data(), cuda_buffers[outputIndex], sizeof_out,
                  cudaMemcpyDeviceToHost, cuda_stream);


  // sync point
  cudaStreamSynchronize(cuda_stream);

  float* slice_p = &output_chw[0];
  output = cv::Mat(cv::Size(w, h), CV_32FC1, slice_p);
  // TODO: crop top bottom
}

}

int main()
{
using namespace bonnet;
Bonnet bnet=Bonnet("bonnetFP32softmax.engine");
cout<<bnet.initialized<<endl;
}
