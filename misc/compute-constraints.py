import numpy as np
import cv2
import utils_box3d
import time
import yaml

image_orig=cv2.imread('kitti.png')
yaw = 0.019480967695382434 #+ np.pi/2
dims = np.array([1.48773544, 1.59376032, 3.74524751])
box_2D = np.array([ 767.,  176., 1084.,  357.], dtype=np.float)
P = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
                  [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]])

inds_right_0_89 = []
inds_right_90_179 = []
inds_right_180_269 = []
inds_right_270_359 = []
inds_left_0_89 = []
inds_left_90_179 = []
inds_left_180_269 = []
inds_left_270_359 = []

# theta
box_2D_orig = box_2D.copy()
for offset in [0, -150]:
  box_2D[0] = box_2D_orig[0] + offset
  box_2D[2] = box_2D_orig[2] + offset
  for yaw in range(0, 89, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_right_0_89.append(new)
    utils_box3d.inds_used = []
  for yaw in range(90, 179, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_right_90_179.append(new)
    utils_box3d.inds_used = []
  for yaw in range(180, 269, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_right_180_269.append(new)
    utils_box3d.inds_used = []
  for yaw in range(270, 359, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_right_270_359.append(new)
    utils_box3d.inds_used = []
for offset in [-500, -750]:
  box_2D[0] = box_2D_orig[0] + offset
  box_2D[2] = box_2D_orig[2] + offset
  for yaw in range(0, 89, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_left_0_89.append(new)
    utils_box3d.inds_used = []
  for yaw in range(90, 179, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_left_90_179.append(new)
    utils_box3d.inds_used = []
  for yaw in range(180, 269, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_left_180_269.append(new)
    utils_box3d.inds_used = []
  for yaw in range(270, 359, 2):
    yaw = yaw * (np.pi/180.0)
    points2D = utils_box3d.gen_3D_box(yaw, dims, P, box_2D)
    for i in utils_box3d.inds_used:
      new = str(i[0])+str(i[1])+str(i[2])+str(i[3])
      inds_left_270_359.append(new)
    utils_box3d.inds_used = []

d={}
d["r0"]   = set(inds_right_0_89)
d["r90"]  = set(inds_right_90_179)
d["r180"] = set(inds_right_180_269)
d["r270"] = set(inds_right_270_359)
d["l0"]   = set(inds_left_0_89)
d["l90"]  = set(inds_left_90_179)
d["l180"] = set(inds_left_180_269)
d["l270"] = set(inds_left_270_359)
with open('result.yml', 'w') as yaml_file:
    yaml.dump(d, yaml_file, default_flow_style=False)
