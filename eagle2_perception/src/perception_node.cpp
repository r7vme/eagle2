#include <ros/ros.h>

#include "perception.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "eagle2_perception");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");
  perception::EPerception p(nh, priv_nh);
  ros::spin();
  return 0;
}
