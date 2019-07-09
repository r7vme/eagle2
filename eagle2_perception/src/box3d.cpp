#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
// Nvidia stuff
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "common.hpp"
#include "box3d.hpp"
#include "l2norm_helper.h"

using namespace std;
using namespace nvinfer1;

namespace box3d
{

Box3D::Box3D(const string& engineFile)
  : input_index(0), ori_index(1), dims_index(2), conf_index(3)
{
  // read engine from file
  fstream file;
  file.open(engineFile, ios::binary | ios::in);
  if(!file.is_open())
  {
      cout<<"box3d: failed to read engine file "<<engineFile<<endl;
      return;
  }
  file.seekg(0, ios::end);
  int length = file.tellg();
  file.seekg(0, ios::beg);
  unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();

  // initialize plugins, required for LeakyRelu
  if (!initLibNvInferPlugins(&logger, ""))
  {
    cout<<"Can not initialize default plugins, "
          "assuming already initialized by other instance."<<endl;
  }

  // create engine
  runtime = createInferRuntime(logger);
  assert(runtime != nullptr);
  engine=runtime->deserializeCudaEngine(data.get(), length, nullptr);
  assert(engine != nullptr);

  // create an execution context
  context = engine->createExecutionContext();
  assert(context != nullptr);

  // allocate memory
  int num_bindings = engine->getNbBindings();
  cuda_buffers.resize(num_bindings);
  cuda_buffers_size.resize(num_bindings);
  for (int i = 0; i < num_bindings; ++i)
  {
      Dims dims = engine->getBindingDimensions(i);
      DataType dtype = engine->getBindingDataType(i);
      int64_t singleSize = get_dims_volume(dims)*get_elem_size(dtype);
      cuda_buffers_size[i] = singleSize;
      cuda_buffers[i] = safeCudaMalloc(MAX_BATCH_SIZE*singleSize);
  }

  // Use CUDA streams to manage the concurrency of copying and executing
  CUDA_CHECK(cudaStreamCreate(&cuda_stream));

  // we are good
  initialized=true;
}

Box3D::~Box3D()
{
  // Release the stream and the buffers
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
  CUDA_CHECK(cudaStreamDestroy(cuda_stream));
  for(auto& item : cuda_buffers)
      CUDA_CHECK(cudaFree(item));
  if(runtime)
      runtime->destroy();
  if(context)
      context->destroy();
  if(engine)
      engine->destroy();
};

void Box3D::doInference(
  int batch_size,
  const vector<float>& input_data,
        vector<float>& out_dims,
        vector<float>& out_ori,
        vector<float>& out_conf)
{
    CUDA_CHECK(cudaMemcpyAsync(cuda_buffers[input_index],
                               input_data.data(),
                               batch_size*cuda_buffers_size[input_index],
                               cudaMemcpyHostToDevice,
                               cuda_stream));

    context->execute(batch_size, &cuda_buffers[input_index]);

    CUDA_CHECK(cudaMemcpyAsync(out_dims.data(),
                               cuda_buffers[dims_index],
                               batch_size*cuda_buffers_size[dims_index],
                               cudaMemcpyDeviceToHost,
                               cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(out_ori.data(),
                               cuda_buffers[ori_index],
                               batch_size*cuda_buffers_size[ori_index],
                               cudaMemcpyDeviceToHost,
                               cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(out_conf.data(),
                               cuda_buffers[conf_index],
                               batch_size*cuda_buffers_size[conf_index],
                               cudaMemcpyDeviceToHost,
                               cuda_stream));
    cudaStreamSynchronize(cuda_stream);
}

}
