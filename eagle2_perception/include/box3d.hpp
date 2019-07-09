#include <stdlib.h>
// Nvidia stuff
#include <cuda.h>
#include <NvInfer.h>

using namespace std;
using namespace nvinfer1;

namespace box3d
{

const int MAX_BATCH_SIZE = 10;
const int C = 3;
const int W = 224;
const int H = 224;
const int BIN_NUM = 6;
const int INPUT_SIZE = C*W*H;
const int DIMS_SIZE = 3;
const int ORI_SIZE = 2*BIN_NUM;
const int CONF_SIZE = BIN_NUM;

class Box3DLogger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
      cout<<"box3d: "<<msg<<endl;
  }
};

class Box3D
{
  public:
    Box3D(const string& engineFile);
    ~Box3D();

    void doInference(int batch_size,
                     const vector<float>& input_data,
                           vector<float>& out_dims,
                           vector<float>& out_ori,
                           vector<float>& out_conf);

    bool initialized=false;
  private:
    int max_batch_size;

    // TensorRT and CUDA related
    Box3DLogger       logger;
    IExecutionContext* context;
    ICudaEngine*       engine;
    IRuntime*          runtime;
    cudaStream_t cuda_stream;
    int input_index;
    int ori_index;
    int dims_index;
    int conf_index;
    vector<void*> cuda_buffers;
    vector<int64_t> cuda_buffers_size;
};

}
