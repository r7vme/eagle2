#include <chrono>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <TrtNet.h>
#include <yaml-cpp/yaml.h>
#include <YoloLayer.h>

#include "utils.hpp"


using namespace std;
using namespace perception;

//tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn) {
//  tensorflow::Status status;
//
//  // Read in the protobuf graph we exported
//  tensorflow::MetaGraphDef graph_def;
//  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
//  if (status != tensorflow::Status::OK()) return status;
//
//  // create the graph in the current session
//  status = sess->Create(graph_def.graph_def());
//  if (status != tensorflow::Status::OK()) return status;
//
//  return tensorflow::Status::OK();
//}

int main()
{
  YAML::Node config = YAML::LoadFile("config.yaml");

  // get config
  int cam_fps                = config["cam_fps"].as<int>();
  int cam_width              = config["cam_width"].as<int>();
  int cam_height             = config["cam_height"].as<int>();
  string cam_id              = config["cam_id"].as<string>();
  string cam_frame_id        = config["cam_frame_id"].as<string>();
  bool do_viz                = config["do_viz"].as<bool>();
  int top_down_width         = config["top_down_width"].as<int>();
  int top_down_height        = config["top_down_height"].as<int>();
  float top_down_resolution  = config["top_down_resolution"].as<float>();

  string yolo_engine = config["yolo_engine"].as<string>();
  string box3d_pb = config["box3d_pb"].as<string>();
  string enet_pb = config["enet_pb"].as<string>();

  // yolo (TensorRT)
  unique_ptr<Tn::trtNet> net;
  // TODO: it can fail silently, check if initialized.
  net.reset(new Tn::trtNet(yolo_engine));

  // tensorflow session options
  //tensorflow::SessionOptions tf_options;
  //tf_options.config.mutable_gpu_options()->set_allow_growth(true);

  //// box3d (TensorFlow)
  //tensorflow::Session *b3d_sess;
  //TF_CHECK_OK(tensorflow::NewSession(tf_options, &b3d_sess));
  //TF_CHECK_OK(LoadModel(b3d_sess, box3d_pb));

  // prepare inputs
  //tensorflow::TensorShape data_shape({1, 2});
  //tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);
  //tensor_dict feed_dict = {
  //    {"input", data},
  //};
  //vector<tensorflow::Tensor> outputs;
  //TF_CHECK_OK(sess->Run(feed_dict, {"output", "dense/kernel:0", "dense/bias:0"},
  //                      {}, &outputs));

  cv::VideoCapture cap(cam_id);
  if (!cap.isOpened())
     return -1;
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);
  cap.set(cv::CAP_PROP_FPS, cam_fps);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
  // only from 4.2
  // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  cv::Mat frame;
  while(true)
  {
    if (!cap.read(frame))
      continue;

    vector<float> input_data = prepare_image(frame);

    int output_count = net->getOutputSize()/sizeof(float);
    unique_ptr<float[]> output_data(new float[output_count]);

    net->doInference(input_data.data(), output_data.get(), 1);

    //Get Output
    auto output = output_data.get();
    //first detect count
    int detCount = output[0];
    //later detect result
    vector<Yolo::Detection> result;
    result.resize(detCount);
    memcpy(result.data(), &output[1], detCount*sizeof(Yolo::Detection));
    auto bboxes = postprocess_image(frame, result, YOLO_NUM_CLS);

    //auto t_start = std::chrono::high_resolution_clock::now();
    //auto t_end = std::chrono::high_resolution_clock::now();
    //float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    //std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    //draw on image
    for(const auto& bbox: bboxes)
    {
      cv::rectangle(
        frame,
        cv::Point(bbox.left,bbox.top),
        cv::Point(bbox.right,bbox.bot),
        cv::Scalar(0,0,255), 3, 8, 0
      );
    }
    //cv::imwrite("result.jpg", frame);
    //cv::imshow("result", frame);
    //cv::waitKey(10);
  }
  return 0;
}
