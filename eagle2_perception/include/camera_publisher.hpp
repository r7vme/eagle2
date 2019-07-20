#pragma once
#include <vector>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

using namespace std;

namespace perception
{

template<typename _Tp> static  cv::Mat toMat(const vector<vector<_Tp> > vecIn) {
    cv::Mat_<_Tp> matOut(vecIn.size(), vecIn.at(0).size());
    for (int i = 0; i < matOut.rows; ++i) {
        for (int j = 0; j < matOut.cols; ++j) {
            matOut(i, j) = vecIn.at(i).at(j);
        }
    }
    return matOut;
}

// Simple OpenCV image publisher.
//
// 1. Read image from OpenCV video capture
// 2. Undistort image.
// 3. Publish
//
class ECameraPublisher
{
public:
  ECameraPublisher(ros::NodeHandle nh, ros::NodeHandle private_nh);

private:
  // ROS parameters
  string config_file_;

  // params
  string cam_id_;
  string cam_frame_id_;
  int cam_width_;
  int cam_height_;
  int cam_fps_;
  double cam_delay_;
  cv::Mat K_;
  cv::Mat D_;

  // undistortGPU related
  cv::Mat xmap_, ymap_;
  cv::cuda::GpuMat xmapG_, ymapG_;
  cv::cuda::GpuMat inputG_, outputG_;

  cv::VideoCapture cap_;
  cv::Mat frame_;
  cv_bridge::CvImage bridge_;

  // ROS services
  image_transport::ImageTransport it_;
  image_transport::Publisher image_pub_;
  ros::Timer timer_;

  // ROS callbacks
  void timerCallback(const ros::TimerEvent& event);

  // helpers
  void undistortGPU();
};

} // namespace perception
