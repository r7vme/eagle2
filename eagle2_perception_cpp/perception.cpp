#include <chrono>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <TrtNet.h>
#include <yaml-cpp/yaml.h>
#include <YoloLayer.h>

#include "utils.hpp"


using namespace std;
using namespace perception;

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
  vector<vector<float>> dims_avg = config["voc_dims_avg"].as<vector<vector<float>>>();

  string yolo_engine = config["yolo_engine"].as<string>();
  string box3d_pb = config["box3d_pb"].as<string>();
  string enet_pb = config["enet_pb"].as<string>();

  // yolo (TensorRT)
  unique_ptr<Tn::trtNet> net;
  // TODO: it can fail silently, check if initialized.
  net.reset(new Tn::trtNet(yolo_engine));

  // tensorflow session options
  tensorflow::SessionOptions tf_options;
  tf_options.config.mutable_gpu_options()->set_allow_growth(true);

  // box3d (TensorFlow)
  tensorflow::Session *b3d_sess;
  TF_CHECK_OK(tensorflow::NewSession(tf_options, &b3d_sess));
  TF_CHECK_OK(LoadModel(b3d_sess, box3d_pb));
  vector<string> b3d_tensors{"input_1","dimension/LeakyRelu",
                             "orientation/l2_normalize", "confidence/Softmax"};

  // enet (TensorFlow)
  tensorflow::Session *enet_sess;
  TF_CHECK_OK(tensorflow::NewSession(tf_options, &enet_sess));
  TF_CHECK_OK(LoadModel(enet_sess, enet_pb));
  vector<string> enet_tensors{"imgs_ph","early_drop_prob_ph", "fullconv/Relu"};

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

    // prepare inputs
    tensorflow::Tensor b3d_input(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, B3D_H, B3D_W, B3D_C}));
    auto b3d_input_mapped = b3d_input.tensor<float, 4>();

    cv::Rect patchRect(0, 0, 100, 100);
    cv::Mat patch;
    cv::resize(frame(patchRect), patch, cv::Size(B3D_W, B3D_H), 0, 0, CV_INTER_CUBIC);
    patch.convertTo(patch, CV_32FC3);
    subtract(patch,NORM_CAFFE,patch);

    {
      const float * source_data = (float*) patch.data;
      // copying the data into the corresponding tensor
      for (int y = 0; y < B3D_H; ++y) {
        const float* source_row = source_data + (y * B3D_W * B3D_C);
        for (int x = 0; x < B3D_W; ++x) {
          const float* source_pixel = source_row + (x * B3D_C);
          for (int c = 0; c < B3D_C; ++c) {
            const float* source_value = source_pixel + c;
            b3d_input_mapped(0, y, x, c) = *source_value;
          }
        }
      }
    }
    vector<tensorflow::Tensor> b3d_output;
    // running the loaded graph
    tensorflow::Status b3d_status  = b3d_sess->Run(
      {{b3d_tensors[0], b3d_input}},
      {b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]},
      {},
      &b3d_output
    );
    if (!b3d_status.ok())
      continue;

    // enet
    // prepare inputs
    tensorflow::Tensor enet_input(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, ENET_H, ENET_W, ENET_C}));
    auto enet_input_mapped = enet_input.tensor<float, 4>();

    cv::Mat enet_in;
    cv::resize(frame, enet_in, cv::Size(ENET_W, ENET_H), 0, 0, CV_INTER_CUBIC);
    enet_in.convertTo(enet_in, CV_32FC3);
    subtract(enet_in,NORM_CITY,enet_in);

    {
      const float * source_data = (float*) enet_in.data;
      // copying the data into the corresponding tensor
      for (int y = 0; y < ENET_H; ++y) {
        const float* source_row = source_data + (y * ENET_W * ENET_C);
        for (int x = 0; x < ENET_W; ++x) {
          const float* source_pixel = source_row + (x * ENET_C);
          for (int c = 0; c < ENET_C; ++c) {
            const float* source_value = source_pixel + c;
            enet_input_mapped(0, y, x, c) = *source_value;
          }
        }
      }
    }
    tensorflow::Tensor enet_input2(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    enet_input2.scalar<float>()() = 0.0;

    vector<tensorflow::Tensor> enet_output;
    tensorflow::Status enet_status  = enet_sess->Run(
      {
        {enet_tensors[0], enet_input},
        {enet_tensors[1], enet_input2}
      },
      {enet_tensors[2]},
      {},
      &enet_output
    );
    if (!enet_status.ok())
      continue;

    cv::Mat img(ENET_H, ENET_W, CV_8UC3);
    label_image_to_color(enet_output[0], img);
    //auto t_start = std::chrono::high_resolution_clock::now();
    //auto t_end = std::chrono::high_resolution_clock::now();
    //float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    //std::cout << "Time taken for inference is " << total << " ms." << std::endl;
    cv::imwrite("1.jpg", img);


    //cout << enet_output[0].DebugString() << endl;
    //cout << enet_output.size() << endl;

    //tensorflow::TensorShape data_shape({1, 2});
    //tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);
    //tensor_dict feed_dict = {
    //    {"input", data},
    //};
    //vector<tensorflow::Tensor> outputs;
    //TF_CHECK_OK(sess->Run(feed_dict, {"output", "dense/kernel:0", "dense/bias:0"},
    //                      {}, &outputs));

    //cv::imwrite("result.jpg", frame);
    //cv::imshow("result", frame);
    //cv::waitKey(10);
  }
  return 0;
}
