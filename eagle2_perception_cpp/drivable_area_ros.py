#!/usr/bin/env python3
import os
import cv2
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import time
import numpy as np
import tensorflow as tf

H = np.array([[3.36202259e-03,1.26885519e-02,-2.34205816e+00],
              [0.00000000e+0,2.53771038e-02,-3.99449824e+00],
              [-0.00000000e+00,5.07542075e-05,-5.98899649e-03]])

bridge = CvBridge()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
sess = tf.Session(graph=graph, config=config)

with tf.gfile.GFile("frozen_model/optimized.pb", 'rb') as f:
    frozen_graph_def = tf.GraphDef()
    frozen_graph_def.ParseFromString(f.read())
with graph.as_default():
    tensors = tf.import_graph_def(frozen_graph_def,
                                  return_elements=["test_model/model/images/truediv:0", "test_model/model/logits/linear/BiasAdd:0"])



def on_image(msg):
    # If already bgr8 passthru encoding.
    if msg.encoding == "8UC3":
        img = bridge.imgmsg_to_cv2(msg)
    else:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    img_orig = img.copy()
    img_orig = cv2.resize(img_orig, (512, 256))
    img = img.astype(np.float32, copy=False)
    img = cv2.resize(img, (512, 256))
    img = (img - 128.) / 128.
    img = np.expand_dims(img, axis=0)
    pred = sess.run(tensors[1],feed_dict={tensors[0]: img})[0]
    pred = np.argmax(pred, axis=2)
    pred = pred.astype(np.uint8)
    with np.nditer(pred, op_flags=['readwrite']) as it:
        for x in it:
            if x == 0:
                x[...] = 0
            else:
                x[...] = 255
    pred = cv2.warpPerspective(pred, H, (500, 500))
    cv2.namedWindow("window1")
    cv2.namedWindow("window2")
    cv2.imshow("window1", pred)
    cv2.imshow("window2", img_orig)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('drivable_area_ros')
    image_sub = rospy.Subscriber("/kitti/camera_color_left/image_rect",
                                 Image,
                                 on_image,
                                 buff_size=8388608,
                                 queue_size=1)
    rospy.spin()
