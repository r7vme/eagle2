#include <boost/shared_ptr.hpp>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include "camera_publisher.hpp"

namespace perception
{

class ECameraPublisherNodelet: public nodelet::Nodelet
{
public:
  ECameraPublisherNodelet() {}

private:
  virtual void onInit(void);
  boost::shared_ptr<ECameraPublisher> cam_pub_;
};

void ECameraPublisherNodelet::onInit()
{
  NODELET_DEBUG("Initializing eagle2_camera_publisher nodelet");
  cam_pub_.reset(new ECameraPublisher(getNodeHandle(), getPrivateNodeHandle()));
}

} // namespace perception

PLUGINLIB_EXPORT_CLASS(perception::ECameraPublisherNodelet, nodelet::Nodelet);
