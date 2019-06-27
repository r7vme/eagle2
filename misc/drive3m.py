#!/usr/bin/env python3
import os
import time
import rospy
from ackermann_msgs.msg import AckermannDriveStamped


if __name__ == '__main__':
    rospy.init_node('drive3m')
    pub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)
    msg = AckermannDriveStamped()
    msg.drive.steering_angle = 0.0
    msg.drive.speed = 0.5
    time.sleep(2)
    t1=time.time()
    pub.publish(msg)
    time.sleep(5)
    msg.drive.speed = 0.0
    pub.publish(msg)
    t2=time.time()
    print("time passed %.2f ms" % (1000*(t2-t1)))
