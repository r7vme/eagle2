#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include "camera_publisher.hpp"

using namespace std;

namespace perception
{

ECameraPublisher::ECameraPublisher(ros::NodeHandle nh, ros::NodeHandle priv_nh): it_(nh)
{
  priv_nh.param<string>("config_file", config_file_, "config.yaml");

  image_pub_ = it_.advertise("image", 1);

  YAML::Node config;
  try
  {
    config = YAML::LoadFile(config_file_);
  }
  catch (const YAML::BadFile&)
  {
    ROS_ERROR_STREAM("Failed to read config file at "<<config_file_);
    ros::shutdown();
    return;
  }

  // get config
  // TODO: catch YAML expection
  cam_id_                     = config["cam_id"].as<string>();
  cam_frame_id_               = config["cam_frame_id"].as<string>();
  cam_width_                  = config["cam_width"].as<int>();
  cam_height_                 = config["cam_height"].as<int>();
  cam_fps_                    = config["cam_fps"].as<int>();
  cam_delay_                  = config["cam_delay"].as<double>();
  vector<vector<float>> K_vec = config["K"].as<vector<vector<float>>>();
  vector<vector<float>> D_vec = config["D"].as<vector<vector<float>>>();

  // prepare CV matrices
  K_ = toMat(K_vec);
  D_ = toMat(D_vec);

  // v4l camera capture. cam_id also can be just a path to a video
  cap_.open(cam_id_);
  if (!cap_.isOpened())
  {
    ros::shutdown();
    return;
  }
  cap_.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap_.set(cv::CAP_PROP_FOCUS, 0);
  cap_.set(cv::CAP_PROP_FPS, cam_fps_);
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, cam_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height_);
  // only from 4.2
  // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  timer_ = nh.createTimer(ros::Duration(1.0/cam_fps_), &ECameraPublisher::timerCallback, this);
}

void ECameraPublisher::undistortGPU()
{
  cv::initUndistortRectifyMap(K_, D_, cv::Mat(), K_,
                              cv::Size(frame_.cols, frame_.rows),
                              CV_32FC1, xmap_, ymap_);
  // copy host mem to device
  inputG_.upload(frame_);
  xmapG_.upload(xmap_);
  ymapG_.upload(ymap_);

  cv::cuda::remap(inputG_, outputG_, xmapG_, ymapG_, cv::INTER_LINEAR);
  bridge_.image = cv::Mat(outputG_);
}

void ECameraPublisher::timerCallback(const ros::TimerEvent& event)
{
  if (!cap_.read(frame_))
  {
    ROS_WARN("Can not read image from cv::VideoCapture.");
    return;
  }
  bridge_.encoding = sensor_msgs::image_encodings::BGR8;
  bridge_.header.stamp = ros::Time::now() - ros::Duration(cam_delay_);
  bridge_.header.frame_id = cam_frame_id_;
  // undistort image
  //cv::undistort(frame_, bridge_.image, K_, D_);
  undistortGPU();
  // publish
  image_pub_.publish(*bridge_.toImageMsg());
}

} // namespace perception
