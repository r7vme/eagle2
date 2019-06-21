#!/usr/bin/env python3
import os
import cv2
import rospy
import yaml
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from scipy.special import softmax
import sys
import time
import numpy as np
import tensorflow as tf
from tf.transformations import euler_from_quaternion

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

def logit(x):
  if x >= 100:
    return sys.float_info.max
  if x <= 0:
    return -sys.float_info.max
  x/=100
  return float(np.log(x/(1-x)))

def inv_logit(l):
  if l==float("inf"):
    return 100
  if l==float("-inf"):
    return 0
  il = 1-(1/(1+np.exp(l)))
  return int(il*100)

def occupancy_grid_mapping(grid_map, grid_update, M):
  gm_h=grid_map.shape[0]
  gm_w=grid_map.shape[1]
  gu_h=grid_update.shape[0]
  gu_w=grid_update.shape[1]
  orig_top_x=gu_h
  orig_top_y=gu_w/2
  for i in range(gu_h):
    for j in range(gu_w):
      x_u=gu_h-(i+orig_top_x)
      y_u=gu_w-(j+orig_top_y)
      x = int(M[0][0]*x_u + M[0][1]*y_u + M[0][2])
      y = int(M[1][0]*x_u + M[1][1]*y_u + M[1][2])
      if not (0<=x<gm_h):
        continue
      if not (0<=y<gm_w):
        continue
      p_cur=grid_map[x][y]
      p_upd=grid_update[i][j]
      if p_upd==255:
        continue
      l_cur=logit(p_cur)
      l_upd=logit(p_upd)
      l_new=l_cur+l_upd
      p_new=inv_logit(l_new)
      grid_map[x][y]=int(p_new)

def get_affine_matrix(theta, tx, ty):
  return np.array([[np.cos(theta), -np.sin(theta), tx],
                   [np.sin(theta), np.cos(theta),  ty]])

# 20cm/px, 200meters x 200meters
grid_map_width = 2000
grid_map_height = 2000
grid_map = np.full((grid_map_height,grid_map_width), 50, dtype=np.uint8)
orig_x=300
orig_y=grid_map_width/2
map_meter_per_pixel=0.2
top_meter_per_pixel=0.04175
top_scale=top_meter_per_pixel/map_meter_per_pixel

# world
# camera_left
i=0

def on_image(msg):
    global i
    # If already bgr8 passthru encoding.
    if msg.encoding == "8UC3":
        img = bridge.imgmsg_to_cv2(msg)
    else:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    img_orig = img.copy()
    img_orig = cv2.resize(img_orig, (512, 256))
    img_orig = cv2.warpPerspective(img_orig, H, (500, 500), borderValue=255)
    img = img.astype(np.float32, copy=False)
    img = cv2.resize(img, (512, 256))
    img = (img - 128.) / 128.
    img = np.expand_dims(img, axis=0)
    pred = sess.run(tensors[1],feed_dict={tensors[0]: img})[0]
    pred = softmax(pred, axis=2)
    pred = np.delete(pred, np.s_[1:20], axis=2)
    with np.nditer(pred, op_flags=['readwrite']) as it:
      for x in it:
        x[...] = x*100
    pred = pred.astype(np.uint8)
    pred = cv2.warpPerspective(pred, H, (500, 500), borderValue=255)
    pred = cv2.copyMakeBorder(pred,0,10,0,0,cv2.BORDER_REFLECT)
    pred=pred[150:,100:400]
    #pred=np.full((300,300), 70, dtype=np.uint8)
    #pred[0:10,:]=0
    #pred[:,105:115]=0
    h=pred.shape[0]
    w=pred.shape[1]
    pred=cv2.resize(pred, (int(w*top_scale), int(h*top_scale)))
    pred_viz=pred.copy()
    pred=cv2.flip(pred,-1)

    trans = tfBuffer.lookup_transform("world", 'camera_left', rospy.Time())
    # ROS coordinates (in meters) -> Occupancy matrix (pixels - 20cm/px)
    # x -> -x
    # y -> -y
    # yaw -> -yaw
    # 20cm/px
    x=grid_map_height-((trans.transform.translation.x/map_meter_per_pixel)+orig_x)
    y=grid_map_width-((trans.transform.translation.y/map_meter_per_pixel)+orig_y)
    q=[trans.transform.rotation.x,
       trans.transform.rotation.y,
       trans.transform.rotation.z,
       trans.transform.rotation.w]
    yaw=euler_from_quaternion(q)[2]
    #x=grid_map_height-((10+i/map_meter_per_pixel)+orig_x)
    #y=grid_map_width-((0/map_meter_per_pixel)+orig_y)
    #yaw=(i*5)*(np.pi/180.)
    #i+=1
    M=get_affine_matrix(yaw, x, y)

    occupancy_grid_mapping(grid_map, pred, M)

    cv2.namedWindow("window1", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("window2", cv2.WINDOW_NORMAL)
    cv2.imshow("window1", grid_map)
    #cv2.imshow("window2", pred_viz)
    cv2.waitKey(10)

if __name__ == '__main__':
    rospy.init_node('drivable_area_ros')
    image_sub = rospy.Subscriber("/kitti/camera_color_left/image_rect",
                                 Image,
                                 on_image,
                                 buff_size=8388608,
                                 queue_size=1)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.spin()
