#!/usr/bin/env python3
import yaml
import sys
import signal
import cv2
import numpy as np
import tf2_ros
import rospy
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion

with open("/home/r7vme/catkin_ws/src/eagle2_perception/cfg/config.yaml", 'r') as f:
    cfg=yaml.load(f)

cam_height=cfg["cam_height_from_ground"]
fy=cfg["K"][1][1]
v0=cfg["K"][1][2]
map_res=cfg["top2_res"]

# 10cm/px, 100meters x 100meters
gmap=OccupancyGrid()
gmap.header.frame_id="map"
gmap.info.resolution=map_res
gmap.info.width=1000
gmap.info.height=500
gmap.info.origin.position.x=-10.0
gmap.info.origin.position.y=-25.0
gmap.info.origin.position.z=0.0
gmap.info.origin.orientation.x=0.0
gmap.info.origin.orientation.y=0.0
gmap.info.origin.orientation.z=0.0
gmap.info.origin.orientation.w=1.0
gmap_data=np.full((500,1000), 50, dtype=np.uint8)

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

def occupancy_grid_mapping(grid_map, grid_update, M, pivot):
  gm_h=grid_map.shape[0]
  gm_w=grid_map.shape[1]
  gu_h=grid_update.shape[0]
  gu_w=grid_update.shape[1]

  # map every point to exitsing map
  for i in range(gu_h):
    for j in range(gu_w):
      x_u=gu_h-(i+pivot[0])
      y_u=gu_w-(j+pivot[1])
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
      # new value is sum of probabilities
      l_cur=logit(p_cur)
      l_upd=logit(p_upd)
      l_new=l_cur+l_upd
      p_new=inv_logit(l_new)
      grid_map[x][y]=int(p_new)

def get_affine_matrix(theta, tx, ty):
  return np.array([[np.cos(theta), -np.sin(theta), tx],
                   [np.sin(theta), np.cos(theta),  ty]])

def on_grid(msg):
    try:
      t = tfBuffer.lookup_transform("map", 'camera', msg.header.stamp)
    except tf2_ros.LookupException:
      rospy.logwarn("Failed to get transform map->camera")
      return

    # ROS coordinates (in meters) -> Occupancy matrix (pixels)
    # x -> y
    # y -> x
    # yaw -> -(yaw + Pi/2)
    y=(t.transform.translation.x-gmap.info.origin.position.x)/gmap.info.resolution
    x=(t.transform.translation.y-gmap.info.origin.position.y)/gmap.info.resolution
    q=[t.transform.rotation.x,
       t.transform.rotation.y,
       t.transform.rotation.z,
       t.transform.rotation.w]
    yaw=-(euler_from_quaternion(q)[2] + np.pi/2)

    # pivot coordinates (camera some distance away from top-down reprojection)
    pivot = (msg.info.width + cam_height*(fy/v0)/map_res, msg.info.width/2)

    # compute affine transformation matrix
    M=get_affine_matrix(yaw, x, y)

    update_data=np.array(msg.data, dtype=np.uint8).reshape(msg.info.height,msg.info.width)

    # update map
    occupancy_grid_mapping(gmap_data, update_data, M, pivot)

    # publish updated map
    #gmap.data=gmap_data.reshape(-1)
    #map_pub.publish(gmap)

def signal_handler(sig, frame):
    print('Saving map to /tmp/map.png!')
    img=gmap_data*(255/100) # scale color from 0..255
    img=(255-img) # invert colors
    img=cv2.flip(img, 0) # invert Y axis
    cv2.imwrite("/tmp/map.png", img)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

rospy.init_node('build_map')
grid_sub=rospy.Subscriber("/drivable_area",OccupancyGrid,on_grid,queue_size=1)
#map_pub=rospy.Publisher("/map",OccupancyGrid)
tfBuffer=tf2_ros.Buffer()
listener=tf2_ros.TransformListener(tfBuffer)
rospy.spin()
