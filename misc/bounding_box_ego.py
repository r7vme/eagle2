#!/usr/bin/env python3
import rospy
from jsk_recognition_msgs.msg import BoundingBox
from tf.transformations import *

rospy.init_node("bbox_ego")
pub = rospy.Publisher("bbox_ego", BoundingBox, queue_size=1)
r = rospy.Rate(24)
counter = 0
while not rospy.is_shutdown():
  now = rospy.Time.now()
  box_a = BoundingBox()
  box_a.header.stamp = now
  box_a.header.frame_id = "camera"
  box_a.pose.orientation.x = 0.0
  box_a.pose.orientation.y = 0.0
  box_a.pose.orientation.z = 0.0
  box_a.pose.orientation.w = 1.0
  box_a.pose.position.x = 0.0
  box_a.pose.position.y = 0.0
  box_a.pose.position.z = 0.5
  box_a.dimensions.x = 1.2
  box_a.dimensions.y = 0.5
  box_a.dimensions.z = 1.0
  pub.publish(box_a)
  r.sleep()
