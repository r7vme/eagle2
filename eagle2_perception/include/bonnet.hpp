#pragma once
#include <stdlib.h>
// Nvidia stuff
#include <cuda.h>
#include <NvInfer.h>

using namespace std;
using namespace nvinfer1;

namespace bonnet
{

const int C=3;
const int W=512;
const int H=256;

class BonnetLogger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
      cout<<"bonnet: "<<msg<<endl;
  }
};

class Bonnet
{
  public:
    Bonnet(const string& engineFile);
    ~Bonnet();

    void doInference(const cv::Mat& image, cv::Mat& output);

    bool initialized=false;
  private:
    unsigned int num_classes, d, w, h;
    // TensorRT and CUDA related
    BonnetLogger       logger;
    IExecutionContext* context;
    ICudaEngine*       engine;
    IRuntime*          runtime;
    int inputIndex, outputIndex; // bindings for cuda i/o
    Dims inputDims, outputDims;  // dimensions of input and output
    int size_in_pix, sizeof_in, sizeof_out;  // size for cuda malloc
    cudaStream_t cuda_stream;  // cuda streams handles copying to/from GPU
    // pointers to GPU memory of input and output
    float* input_gpu;
    int* output_gpu;
    void* cuda_buffers[2];
};

}
