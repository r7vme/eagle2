#include <ros/ros.h>

#include "camera_publisher.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "eagle2_camera_publisher");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");
  perception::ECameraPublisher c(nh, priv_nh);
  ros::spin();
  return 0;
}
