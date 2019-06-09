#pragma once
#include <opencv2/opencv.hpp>
#include <TrtNet.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <YoloLayer.h>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

namespace perception
{

  // box3d utils
  typedef Matrix<float,8,3> Points3D;
  Points3D init_points3D(const std::array<float, 3> &dims);

  // tensorflow utils
  tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn);

  // yolo utils
  struct YoloBbox
  {
      int classId;
      int left;
      int right;
      int top;
      int bot;
      float score;
  };
  const float NORM_CITY[3] = { 72.78044, 83.21195, 73.45286};
  const cv::Scalar NORM_CAFFE(103.939, 116.779, 123.68);
  const map<int, int> COCO_TO_VOC = {{2,0},{5,1},{7,2},{0,3},{1,5},{6,6}};
  const int YOLO_C = 3;
  const int YOLO_W = 416;
  const int YOLO_H = 416;
  const int YOLO_NUM_CLS = 80;
  const float YOLO_NMS_THRESH = 0.45;
  vector<float> prepare_image(cv::Mat& img);
  void do_nms(vector<Yolo::Detection>& detections, int classes, float nmsThresh);
  vector<YoloBbox> postprocess_image(cv::Mat& img, vector<Yolo::Detection>& detections, int classes);
  vector<string> split(const string& str, char delim);

  // box3d related
  // Indicies for 8 specific cases.
  // - R/L means object on the right or left of the image
  // - 0/90/180/270 means object yaw range
  // These numbers were computed from 1024 possible constrainer cases.
  // Main problem was performance as computation even for 256 cases
  // was taking 50ms on core i5.
  const int IND_R0[10][4] =
  {
    {4,0,1,2},
    {4,0,3,2},
    {4,0,7,0},
    {4,0,7,2},
    {4,0,7,4},
    {4,0,7,6},
    {4,2,1,2},
    {4,2,7,2},
    {6,2,1,2},
    {6,2,7,2}
  };
  const int IND_R90[6][4] =
  {
    {2,0,5,2},
    {2,0,5,6},
    {2,6,5,0},
    {2,6,5,6},
    {4,0,5,4},
    {4,0,5,6},
  };
  const int IND_R180[10][4] =
  {
    {0,4,3,0},
    {0,4,3,2},
    {0,4,3,4},
    {0,4,3,6},
    {0,4,5,6},
    {0,4,7,6},
    {0,6,3,0},
    {0,6,3,6},
    {2,6,3,0},
    {2,6,3,6},
  };
  const int IND_R270[6][4] =
  {
    {0,4,1,0},
    {0,4,1,2},
    {6,2,1,2},
    {6,2,1,4},
    {6,4,1,2},
    {6,4,1,6},
  };
  const int IND_L0[8][4] =
  {
    {2,0,7,0},
    {2,0,7,4},
    {2,6,7,4},
    {2,6,7,6},
    {4,0,1,2},
    {4,0,3,2},
    {4,0,7,2},
    {4,0,7,4},
  };
  const int IND_L90[8][4] =
  {
    {0,4,1,0},
    {0,4,5,0},
    {0,4,5,6},
    {0,6,5,0},
    {2,6,5,0},
    {2,6,5,2},
    {2,6,5,4},
    {2,6,5,6},
  };
  const int IND_L180[7][4] =
  {
    {0,4,3,0},
    {0,4,3,6},
    {0,4,7,6},
    {6,2,3,0},
    {6,2,3,2},
    {6,4,3,0},
    {6,4,3,4},
  };
  const int IND_L270[8][4] =
  {
    {4,0,1,2},
    {4,0,1,4},
    {4,0,5,4},
    {4,2,1,4},
    {6,2,1,0},
    {6,2,1,2},
    {6,2,1,4},
    {6,2,1,6},
  };
} // namespace perception
