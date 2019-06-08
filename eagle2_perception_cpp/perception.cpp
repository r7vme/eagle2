#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <TrtNet.h>
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

  string yolo_engine = config["yolo_engine"].as<string>();

  // yolo
  unique_ptr<Tn::trtNet> net;
  // TODO: now it fails silently, check if initialized.
  net.reset(new Tn::trtNet(yolo_engine));

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
      break;

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
          cv::rectangle(frame,cv::Point(bbox.left,bbox.top),cv::Point(bbox.right,bbox.bot),cv::Scalar(0,0,255),3,8,0);
    }
    //cv::imwrite("result.jpg", frame);
  }
  return 0;
}
