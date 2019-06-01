# original version https://github.com/cersar/3D_detection

import time
import cv2
import numpy as np


# Indicies for 8 specific cases.
# - R/L means object on the right or left of the image
# - 0/90/180/270 means object yaw range
# These numbers were computed from 1024 possible constrainer cases.
# Main problem was performance as computation even for 256 cases
# was taking 50ms on core i5.
IND_R0 = [
  [4,0,1,2],
  [4,0,3,2],
  [4,0,7,0],
  [4,0,7,2],
  [4,0,7,4],
  [4,0,7,6],
  [4,2,1,2],
  [4,2,7,2],
  [6,2,1,2],
  [6,2,7,2],
]
IND_R90 = [
  [2,0,5,2],
  [2,0,5,6],
  [2,6,5,0],
  [2,6,5,6],
  [4,0,5,4],
  [4,0,5,6],
]
IND_R180 = [
  [0,4,3,0],
  [0,4,3,2],
  [0,4,3,4],
  [0,4,3,6],
  [0,4,5,6],
  [0,4,7,6],
  [0,6,3,0],
  [0,6,3,6],
  [2,6,3,0],
  [2,6,3,6],
]
IND_R270 = [
  [0,4,1,0],
  [0,4,1,2],
  [6,2,1,2],
  [6,2,1,4],
  [6,4,1,2],
  [6,4,1,6],
]
IND_L0 = [
  [2,0,7,0],
  [2,0,7,4],
  [2,6,7,4],
  [2,6,7,6],
  [4,0,1,2],
  [4,0,3,2],
  [4,0,7,2],
  [4,0,7,4],
]
IND_L90 = [
  [0,4,1,0],
  [0,4,5,0],
  [0,4,5,6],
  [0,6,5,0],
  [2,6,5,0],
  [2,6,5,2],
  [2,6,5,4],
  [2,6,5,6],
]
IND_L180 = [
  [0,4,3,0],
  [0,4,3,6],
  [0,4,7,6],
  [6,2,3,0],
  [6,2,3,2],
  [6,4,3,0],
  [6,4,3,4],
]
IND_L270 = [
  [4,0,1,2],
  [4,0,1,4],
  [4,0,5,4],
  [4,2,1,4],
  [6,2,1,0],
  [6,2,1,2],
  [6,2,1,4],
  [6,2,1,6],
]

COCO_TO_VOC = {2:0,5:1,7:2,0:3,1:5,6:6}

# camera : x->right, y->down, z->forward
# world: x->forward, y->left, z->up
to_world_frame = np.array([
  [  0.,  0.,  1.],
  [ -1.,  0.,  0.],
  [  0., -1.,  0.]
])
to_cam_frame = to_world_frame.T

def compute_yaw(prediction, xmin, xmax, fx, u0):
    theta_ray = np.arctan2(fx, ((xmin + xmax) / 2.0 - u0))
    max_anc = np.argmax(prediction[2][0])
    anchors = prediction[1][0][max_anc]
    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])
    bin_num = prediction[2][0].shape[0]
    wedge = 2. * np.pi / bin_num
    theta_loc = angle_offset + max_anc * wedge
    theta = theta_loc + theta_ray
    # object's yaw angle
    yaw = np.pi/2 - theta
    return yaw, theta_ray


def init_points3D(dims):
    points3D = np.zeros((8, 3))
    cnt = 0
    for i in [1, -1]:
        for j in [1, -1]:
            for k in [1, -1]:
                points3D[cnt] = dims[[1, 0, 2]].T / 2.0 * [i, k, j * i]
                cnt += 1
    return points3D


def points3D_to_2D(points3D,center,rot_M,cam_to_img):
    points2D = []
    for point3D in points3D:
        point3D = point3D.reshape((-1,1))
        point = center + np.dot(rot_M, point3D)
        point = np.append(point, 1)
        point = np.dot(cam_to_img, point)
        point2D = point[:2] / point[2]
        points2D.append(point2D)
    points2D = np.asarray(points2D)

    return points2D


def compute_error(points3D,center,rot_M, cam_to_img,box_2D):
    points2D = points3D_to_2D(points3D, center, rot_M, cam_to_img)
    new_box_2D = np.asarray([np.min(points2D[:,0]),
                  np.max(points2D[:,0]),
                  np.min(points2D[:,1]),
                  np.max(points2D[:,1])]).reshape((-1,1))
    error = np.sum(np.abs(new_box_2D - box_2D))
    return error


def compute_center(points3D,rot_M,cam_to_img,box_2D, inds):
    fx = cam_to_img[0][0]
    fy = cam_to_img[1][1]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]
    W = np.array([[fx, 0, float(u0 - box_2D[0])],
                  [fx, 0, float(u0 - box_2D[2])],
                  [0, fy, float(v0 - box_2D[1])],
                  [0, fy, float(v0 - box_2D[3])]])
    U, Sigma, VT = np.linalg.svd(W)

    center =None
    error_min = 1e10

    for ind in inds:
        y = np.zeros((4, 1))
        for i in range(len(ind)):
            RP = np.dot(rot_M, (points3D[ind[i]]).reshape((-1, 1)))
            # XXX: i'm not 100% understand what is Y
            y[i] = box_2D[i] * cam_to_img[2, 3] - np.dot(W[i], RP) - cam_to_img[i // 2, 3]
        result = np.dot(np.dot(np.dot(VT.T, np.linalg.pinv(np.eye(4, 3) * Sigma)), U.T), y)
        error = compute_error(points3D, result, rot_M, cam_to_img, box_2D)
        if error < error_min and result[2,0]>0:
            center = result
            error_min = error
    return center


def draw_3D_box(image,points):
    points = points.astype(np.int)

    for i in range(4):
        point_1_ = points[2 * i]
        point_2_ = points[2 * i + 1]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 2)

    cv2.line(image,tuple(points[0]),tuple(points[7]),(0, 0, 255), 2)
    cv2.line(image, tuple(points[1]), tuple(points[6]), (0, 0, 255), 2)

    for i in range(8):
        point_1_ = points[i]
        point_2_ = points[(i + 2) % 8]
        cv2.line(image, (point_1_[0], point_1_[1]), (point_2_[0], point_2_[1]), (0, 255, 0), 2)


def draw_2D_box(image,points):
    points = points.astype(np.int)
    cv2.rectangle(image,tuple(points[0:2]),tuple(points[2:4]),(0, 255, 0), 2)


def gen_3D_box(yaw,theta_ray,dims,cam_to_img,box_2D):
    dims = dims.reshape((-1,1))
    box_2D = box_2D.reshape((-1,1))
    points3D = init_points3D(dims)

    rot_M = np.asarray([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    if not (0.<theta_ray<np.pi):
        return None
    if not (-2*np.pi<yaw<2*np.pi):
        return None

    norm_yaw=yaw
    if yaw < 0:
      norm_yaw=yaw+2*np.pi

    if theta_ray < np.deg2rad(90):
      if np.deg2rad(0)<norm_yaw<np.deg2rad(89):
        constraints = IND_R0
      elif np.deg2rad(89)<norm_yaw<np.deg2rad(179):
        constraints = IND_R90
      elif np.deg2rad(179)<norm_yaw<np.deg2rad(269):
        constraints = IND_R180
      else:
        constraints = IND_R270
    else:
      if np.deg2rad(0)<norm_yaw<np.deg2rad(89):
        constraints = IND_L0
      elif np.deg2rad(89)<norm_yaw<np.deg2rad(179):
        constraints = IND_L90
      elif np.deg2rad(179)<norm_yaw<np.deg2rad(269):
        constraints = IND_L180
      else:
        constraints = IND_L270

    center = compute_center(points3D, rot_M, cam_to_img, box_2D, constraints)

    pts_projected = points3D_to_2D(points3D, center, rot_M, cam_to_img)
    pts_world     = points3D_to_world(points3D, center, rot_M, cam_to_img)

    return pts_projected, pts_world


def filter_only_voc_class(bboxes):
    bboxes_copy = bboxes.copy()
    bboxes_new = []
    for b in bboxes_copy:
        cls = int(b[5])
        if cls in COCO_TO_VOC:
            b[5] = COCO_TO_VOC[cls]
            bboxes_new.append(b)
    return bboxes_new

def points3D_to_world(points3D,center,rot_M,cam_to_img):
    points3D_new = np.empty_like(points3D)
    for i, p in enumerate(points3D):
        p = p.reshape((-1,1))
        p = center + np.dot(rot_M, p)
        p = to_world_frame.dot(p)
        points3D_new[i] = p[:,0]
    return points3D_new

def rot_from_euler(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R
