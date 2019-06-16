#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "opencv2/opencv.hpp"
// Nvidia stuff
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvUffParser.h"
#include <NvInferPlugin.h>

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
      std::cout << "logger motherfuckers:" << msg << std::endl;
  }
} logger;


int main()
{
  // Cuda and TensorRT stuff
  IBuilder* builder;            // builder for TensorRT engine
  INetworkDefinition* network;  // where to populate network
  IUffParser* parser;  // parser for uff model (from frozen pb in py)
  ICudaEngine* engine;  // cuda engine to run the model
  IExecutionContext* context;         // context to launch the kernels
  IRuntime* runtime;
  int inputIndex, outputIndex;              // bindings for cuda i/o
  Dims inputDims, outputDims;        // dimensions of input and output
  int size_in_pix, sizeof_in, sizeof_out;  // size for cuda malloc
  cudaStream_t cuda_stream;  // cuda streams handles copying to/from GPU

  // pointers to GPU memory of input and output
  float* input_gpu;
  int* output_gpu;
  void* cuda_buffers[2];

  // graph nodes for i/o
  std::string input_node = "test_model/model/images/truediv";
  std::string output_node = "test_model/model/logits/linear/BiasAdd";
  unsigned int num_classes, d, w, h;
  num_classes = 20;
  d = 3; w = 512; h = 256;

  // read engine from file
  fstream file;
  file.open("bonnetFP32.engine",ios::binary | ios::in);
  if(!file.is_open())
      return -1;
  file.seekg(0, ios::end);
  int length = file.tellg();
  file.seekg(0, ios::beg);
  std::unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();

  bool ok = initLibNvInferPlugins(&logger, "");
  if (!ok) { return 1; }

  std::cout << "deserializing" << std::endl;
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

  cv::VideoCapture cap("kitti_02.avi");
  if (!cap.isOpened())
     return -1;

  cv::Mat image;
  while(true)
  {
  if (!cap.read(image))
    continue;

  cv::Mat norm_image;
  cv::resize(image, norm_image, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
  norm_image.convertTo(norm_image, CV_32FC3);
  norm_image = (norm_image - 128.0f) / 128.0f;

  std::vector<cv::Mat> norm_image_chw(d);
  cv::split(norm_image, norm_image_chw);
  // copy the B, G and, R data into a contiguous array
  std::vector<float> norm_image_chw_data;
  for (unsigned int ch = 0; ch < d; ++ch) {
    if (norm_image_chw[ch].isContinuous()) {
      norm_image_chw_data.insert(norm_image_chw_data.end(),
                                 (float*)norm_image_chw[ch].datastart,
                                 (float*)norm_image_chw[ch].dataend);
    } else {
      for (unsigned int y = 0; y < h; ++y) {
        norm_image_chw_data.insert(norm_image_chw_data.end(),
                                   norm_image_chw[ch].ptr<float>(y),
                                   norm_image_chw[ch].ptr<float>(y) + w);
      }
    }
  }

  auto t_start = std::chrono::high_resolution_clock::now();
  // Copy Input Data to the GPU memory
  cudaMemcpyAsync(cuda_buffers[inputIndex], norm_image_chw_data.data(),
                  sizeof_in, cudaMemcpyHostToDevice, cuda_stream);


  // Enqueue the op
  context->enqueue(1, cuda_buffers, cuda_stream, nullptr);

  // Copy Output Data to the CPU memory
  std::vector<float> output_chw(size_in_pix * num_classes);
  cudaMemcpyAsync(output_chw.data(), cuda_buffers[outputIndex], sizeof_out,
                  cudaMemcpyDeviceToHost, cuda_stream);


  // sync point
  cudaStreamSynchronize(cuda_stream);
  auto t_end = std::chrono::high_resolution_clock::now();
  float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  std::cout << "Time taken for inference is " << total << " ms." << std::endl;

  std::vector<cv::Mat> output_cMats(num_classes);
  cv::Mat output;  // merged output (easier and faster argmax)
  for (unsigned int c = 0; c < num_classes; ++c) {
    float* slice_p = &output_chw[c * size_in_pix];
    output_cMats[c] = cv::Mat(cv::Size(w, h), CV_32FC1, slice_p);
  }
  cv::merge(output_cMats, output);

  // for each pixel, calculate the argmax
  cv::Mat argmax(cv::Size(w, h), CV_32SC1);
  for (unsigned int y = 0; y < h; ++y) {
    int* row_argmax = argmax.ptr<int>(y);
    float* row_c = output.ptr<float>(y);
    for (unsigned int x = 0; x < w; ++x) {
      float max = row_c[x * num_classes];
      int max_c = 0;
      for (unsigned int ch = 1; ch < num_classes; ++ch) {
        if (row_c[x * num_classes + ch] > max) {
          max = row_c[x * num_classes + ch];
          max_c = ch;
        }
      }
      row_argmax[x] = max_c;
      if (max_c==0)
        row_argmax[x] = 255;
    }
  }
  // convert to 8U and put in mask
  cv::Mat mask;
  argmax.convertTo(mask, CV_8U);
  cv::namedWindow("window",CV_WINDOW_AUTOSIZE);
  cv::imshow("window", mask);
  cv::waitKey(1);
  }
  return 0;
}
//
///**
// * @brief      Infer mask from image
// *
// * @param[in]  image    The image to process
// * @param[out] mask     The mask output as argmax of probabilities
// * @param[in]  verbose  Verbose mode (Output timing)
// *
// * @return     Exit code
// */
//retCode NetTRT::infer(const cv::Mat& image, cv::Mat& mask, const bool verbose) {
//  // Check if image has something
//  if (!image.data) {
//    std::cout << "Could find content in the image" << std::endl;
//    return CNN_FAIL;
//  }
//
//  // start total counter
//  auto start_time_total = std::chrono::high_resolution_clock::now();
//
//  // Get dimensions and check that it has proper channels
//  static unsigned int num_classes = _cfg_data["label_remap"].size();
//  static unsigned int d = _cfg_data["img_prop"]["depth"].as<unsigned int>();
//  static unsigned int w = _cfg_data["img_prop"]["width"].as<unsigned int>();
//  static unsigned int h = _cfg_data["img_prop"]["height"].as<unsigned int>();
//  unsigned int cv_img_d = image.channels();
//  assert(cv_img_d == d);
//
//  // Set up inputs to run the graph
//  // First convert to proper size, format, and normalize
//  cv::Mat norm_image;
//  cv::resize(image, norm_image, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
//  norm_image.convertTo(norm_image, CV_32FC3);
//  norm_image = (norm_image - 128.0f) / 128.0f;
//
//  // WATCH OUT! CUDA takes channel first, opencv is channel last
//  // split in B, G, R
//  std::vector<cv::Mat> norm_image_chw(d);
//  cv::split(norm_image, norm_image_chw);
//  // copy the B, G and, R data into a contiguous array
//  std::vector<float> norm_image_chw_data;
//  for (unsigned int ch = 0; ch < d; ++ch) {
//    if (norm_image_chw[ch].isContinuous()) {
//      norm_image_chw_data.insert(norm_image_chw_data.end(),
//                                 (float*)norm_image_chw[ch].datastart,
//                                 (float*)norm_image_chw[ch].dataend);
//    } else {
//      for (unsigned int y = 0; y < h; ++y) {
//        norm_image_chw_data.insert(norm_image_chw_data.end(),
//                                   norm_image_chw[ch].ptr<float>(y),
//                                   norm_image_chw[ch].ptr<float>(y) + w);
//      }
//    }
//  }
//
//  // Run the graph
//  // start inference counter
//  auto start_time_inference = std::chrono::high_resolution_clock::now();
//
//  // Copy Input Data to the GPU memory
//  cudaMemcpyAsync(_cuda_buffers[_inputIndex], norm_image_chw_data.data(),
//                  _sizeof_in, cudaMemcpyHostToDevice, _cuda_stream);
//
//  // Enqueue the op
//  _context->enqueue(1, _cuda_buffers, _cuda_stream, nullptr);
//
//  // Copy Output Data to the CPU memory
//  std::vector<float> output_chw(_size_in_pix * num_classes);
//  cudaMemcpyAsync(output_chw.data(), _cuda_buffers[_outputIndex], _sizeof_out,
//                  cudaMemcpyDeviceToHost, _cuda_stream);
//
//  // sync point
//  cudaStreamSynchronize(_cuda_stream);
//
//  // elapsed_inference time
//  auto elapsed_inference =
//      std::chrono::duration_cast<std::chrono::milliseconds>(
//          std::chrono::high_resolution_clock::now() - start_time_inference)
//          .count();
//
//  // Process the output with map
//  // WATCH OUT! CUDA gives channel first, opencv is channel last
//  // Convert to vector of unidimensional mats and merge
//
//  std::vector<cv::Mat> output_cMats(num_classes);
//  cv::Mat output;  // merged output (easier and faster argmax)
//  for (unsigned int c = 0; c < num_classes; ++c) {
//    float* slice_p = &output_chw[c * _size_in_pix];
//    output_cMats[c] = cv::Mat(cv::Size(w, h), CV_32FC1, slice_p);
//  }
//  cv::merge(output_cMats, output);
//
//  // for each pixel, calculate the argmax
//  cv::Mat argmax(cv::Size(w, h), CV_32SC1);
//  for (unsigned int y = 0; y < h; ++y) {
//    int* row_argmax = argmax.ptr<int>(y);
//    float* row_c = output.ptr<float>(y);
//    for (unsigned int x = 0; x < w; ++x) {
//      float max = row_c[x * num_classes];
//      int max_c = 0;
//      for (unsigned int ch = 1; ch < num_classes; ++ch) {
//        if (row_c[x * num_classes + ch] > max) {
//          max = row_c[x * num_classes + ch];
//          max_c = ch;
//        }
//      }
//      row_argmax[x] = max_c;
//    }
//  }
//
//  // elapsed_total time
//  auto elapsed_total =
//      std::chrono::duration_cast<std::chrono::milliseconds>(
//          std::chrono::high_resolution_clock::now() - start_time_total)
//          .count();
//
//  if (verbose) {
//    std::cout << "Successfully run prediction from engine." << std::endl;
//    std::cout << "Time to infer: " << elapsed_inference << "ms." << std::endl;
//    std::cout << "Time in total: " << elapsed_total << "ms." << std::endl;
//  }
//
//  // convert to 8U and put in mask
//  argmax.convertTo(mask, CV_8U);
//
//  return CNN_OK;
//}
//
///**
// * @brief      Set verbosity level for backend execution
// *
// * @param[in]  verbose  True is max verbosity, False is no verbosity.
// *
// * @return     Exit code.
// */
//retCode NetTRT::verbosity(const bool verbose) {
//  _logger.set_verbosity(verbose);
//  return CNN_OK;
//}
//
//} /* namespace bonnet */
