#pragma once
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <TrtNet.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/OccupancyGrid.h>
#include "bonnet.hpp"
#include "box3d.hpp"

using namespace std;

namespace perception
{

// Core perception module that runs following pipeline:
//
// 1. Receives image from topic
// 2. Yolo v3 obect detection
// 3. 3DBox estimation (only for cars now).
// 4. Drivable area segmentation (used in local planner and mapping).
// 5. Publishes bboxes and occupancy grid (+ vizualization image).
//
class EPerception
{
public:
  EPerception(ros::NodeHandle nh, ros::NodeHandle private_nh);

private:
  // ROS parameters
  string config_file_;

  // params
  string cam_frame_id_;
  int cam_width_;
  int cam_height_;
  float cam_height_from_ground_;
  int top_width_;
  int top_height_;
  float top_res_;
  float top2_res_;
  bool do_viz_;
  cv::Mat H_;
  cv::Mat K_;
  float fx_, fy_, u0_, v0_;
  Eigen::Matrix<float,3,4> P_;
  int top2_width_;
  int top2_height_;
  float cam_origin_x_;
  float cam_origin_y_;
  float bonnet_scale_;

  unique_ptr<Tn::trtNet> yolo_;
  unique_ptr<bonnet::Bonnet> bonnet_net_;
  unique_ptr<box3d::Box3D> box3d_net_;
  cv::Mat frame_viz_;
  cv::Mat bonnet_output_;
  cv_bridge::CvImageConstPtr bridge_;
  nav_msgs::OccupancyGrid map_msg_;

  // ROS services
  ros::Publisher image_pub_;
  ros::Publisher map_pub_;
  ros::Publisher bbox_pub_;
  ros::Publisher viz_pub_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

  // ROS callbacks
  void imageCallback(const sensor_msgs::ImageConstPtr& cmd);
};

} // namespace perception
