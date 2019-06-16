#!/usr/bin/env python3
import graphsurgeon as gs
import tensorflow as tf
import tensorrt as trt
import uff

if __name__ == "__main__":
  data_type = trt.DataType.HALF
  #data_type = trt.DataType.FLOAT
  output_node = "test_model/model/logits/linear/BiasAdd"
  input_node = "test_model/model/images/truediv"
  graph_pb = "optimized_tRT.pb"
  engine_file = "sample.engine"
  dynamic_graph = gs.DynamicGraph(graph_pb)

  # replace LeakyRelu wiht LReLU_TRT plugin
  nodes=[n.name for n in dynamic_graph.as_graph_def().node]
  ns={}
  for node in nodes:
    if "LeakyRelu" in node:
      ns[node]=gs.create_plugin_node(name=node,op="LReLU_TRT", negSlope=0.1)
  dynamic_graph.collapse_namespaces(ns)
  # convert to UFF
  uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), output_nodes=[output_node])

  # convert to TRT
  G_LOGGER = trt.Logger(trt.Logger.ERROR)
  trt.init_libnvinfer_plugins(G_LOGGER, "")
  builder = trt.Builder(G_LOGGER)
  builder.max_batch_size = 1
  builder.max_workspace_size = 1 << 20
  network = builder.create_network()
  parser = trt.UffParser()
  parser.register_input(input_node, trt.Dims([3, 256, 512]))
  parser.register_output(output_node)
  parser.parse_buffer(uff_model, network, data_type)
  engine = builder.build_cuda_engine(network)
  with open(engine_file, "wb") as f:
    f.write(engine.serialize())
