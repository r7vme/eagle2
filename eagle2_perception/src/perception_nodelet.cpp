#include <boost/shared_ptr.hpp>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include "perception.hpp"

namespace perception
{

class EPerceptionNodelet: public nodelet::Nodelet
{
public:
  EPerceptionNodelet() {}

private:
  virtual void onInit(void);
  boost::shared_ptr<EPerception> perception_;
};

void EPerceptionNodelet::onInit()
{
  NODELET_DEBUG("Initializing eagle2_perception nodelet");
  perception_.reset(new EPerception(getNodeHandle(), getPrivateNodeHandle()));
}

} // namespace perception

PLUGINLIB_EXPORT_CLASS(perception::EPerceptionNodelet, nodelet::Nodelet);
