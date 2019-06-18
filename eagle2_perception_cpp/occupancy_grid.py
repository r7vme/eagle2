#!/usr/bin/env python3
import sys
import cv2
import numpy as np

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

def occupancy_grid_mapping(grid_map, grid_update, orig_x, orig_y):
  for i in range(grid_update.shape[0]):
    for j in range(grid_update.shape[1]):
      p_cur=grid_map[orig_x+i][orig_y+j]
      p_upd=grid_update[i][j]
      l_cur=logit(p_cur)
      l_upd=logit(p_upd)
      l_new=l_cur+l_upd
      p_new=inv_logit(l_new)
      grid_map[orig_x+i][orig_y+j]=int(p_new)

# 20cm/px, 200meters x 200meters
grid_map = np.full((1000,1000), 50, dtype=np.uint8)
for i in range(10):
  shift = i*10
  delta = 0
  # 20cm/px, 10meters x 10meters
  grid_update = np.full((50, 50), 53+delta, dtype=np.uint8)
  orig_x, orig_y = 100, 100+shift
  occupancy_grid_mapping(grid_map, grid_update, orig_x, orig_y)
  cv2.namedWindow("window")
  cv2.imshow("window", grid_map)
  cv2.waitKey(100)
