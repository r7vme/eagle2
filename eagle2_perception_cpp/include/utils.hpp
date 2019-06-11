#pragma once
#include <map>
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

  const int TOP_W = 500;
  const int TOP_H = 500;
  const float TOP_RESOLUTION = 0.1; // 10cm/px
  const int CAM_W = 1226;
  const int CAM_H = 370;
  const int CAM_FPS = 10;

  // box3d
  typedef Matrix<float,8,3> Points3D;
  Points3D init_points3D(const std::array<float, 3> &dims);
  const int B3D_MAX_OBJECTS = 4;
  const int B3D_C = 3;
  const int B3D_W = 224;
  const int B3D_H = 224;
  const int B3D_BIN_NUM = 6;
  const map<int, int> COCO_TO_VOC = {{2,0},{5,1},{7,2},{0,3},{1,5},{6,6}};
  struct Bbox3D
  {
      float cos;
      float sin;
      float h;
      float w;
      float l;
      int bin_id;
  };

  // enet
  const int ENET_C = 3;
  const int ENET_W = 1024;
  const int ENET_H = 512;
  const int ENET_NUM_CLS = 20;
  const int ENET_ROAD_CLS = 0;
  const vector<cv::Vec3b> ENET_LABEL_TO_COLOR =
  {
    {128, 64,128},
    {244, 35,232},
    { 70, 70, 70},
    {102,102,156},
    {190,153,153},
    {153,153,153},
    {250,170, 30},
    {220,220,  0},
    {107,142, 35},
    {152,251,152},
    { 70,130,180},
    {220, 20, 60},
    {255,  0,  0},
    {  0,  0,142},
    {  0,  0, 70},
    {  0, 60,100},
    {  0, 80,100},
    {  0,  0,230},
    {119, 11, 32},
    {81,  0,  81}
  };
  void label_image_to_color(const tensorflow::Tensor tr, cv::Mat &img);
  cv::Mat1b get_top_down_occupancy_array(const tensorflow::Tensor tr, const cv::Mat& H);

  // tensorflow utils
  tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn);

  // yolo
  struct YoloBbox
  {
      int classId;
      int left;
      int right;
      int top;
      int bot;
      float score;
  };
  const cv::Scalar NORM_CITY(72.78044, 83.21195, 73.45286);
  const cv::Scalar NORM_CAFFE(103.939, 116.779, 123.68);
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

  // VOC dataset average dimentions
  const float DIMS_AVG[7][3] =
  {
    {1.52131309, 1.64441358, 3.85728004},
    {2.18560847, 1.91077601, 5.08042328},
    {3.07044968, 2.62877944, 11.17126338},
    {1.75562272, 0.67027992, 0.87397566},
    {1.28627907, 0.53976744, 0.96906977},
    {1.73456498, 0.58174006, 1.77485499},
    {3.56020305, 2.40172589, 18.60659898},
  };

  template<typename _Tp> static  cv::Mat toMat(const vector<vector<_Tp> > vecIn) {
      cv::Mat_<_Tp> matOut(vecIn.size(), vecIn.at(0).size());
      for (int i = 0; i < matOut.rows; ++i) {
          for (int j = 0; j < matOut.cols; ++j) {
              matOut(i, j) = vecIn.at(i).at(j);
          }
      }
      return matOut;
  }
} // namespace perception
