# original version https://github.com/cersar/3D_detection

import cv2
import colorsys
import random
import time

import numpy as np
import tensorflow as tf


norm_city = np.array((72.78044, 83.21195, 73.45286))
norm_caffe = np.array([[[103.939, 116.779, 123.68]]])

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

def reproject_point(x, y, H):
    x_bar = (H[0][0]*x + H[0][1]*y + H[0][2]) / \
            (H[2][0]*x + H[2][1]*y + H[2][2])
    y_bar = (H[1][0]*x + H[1][1]*y + H[1][2]) / \
            (H[2][0]*x + H[2][1]*y + H[2][2])
    return (x_bar, y_bar)


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
    pts_projected = pts_projected.astype(np.int)
    return pts_projected


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

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],      
        2: [ 70, 70, 70],      
        3: [102,102,156],      
        4: [190,153,153],      
        5: [153,153,153],      
        6: [250,170, 30],      
        7: [220,220,  0],      
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],     
        11: [220, 20, 60],     
        12: [255,  0,  0],     
        13: [  0,  0,142],     
        14: [  0,  0, 70],
        15: [  0, 60,100],     
        16: [  0, 80,100],     
        17: [  0,  0,230],     
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):   
        for col in range(img_width):    
            label = img[row, col]           
            img_color[row, col] = np.array(label_to_color[label]) 
    return img_color

def read_class_names():
    '''loads class name from a file'''
    class_file_name = 'coco.names'
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_size, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32, copy=False)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = cv2.resize(image, (nw, nh))
    image_paded = image_paded / 255.
    return image_paded


def draw_bbox(image, bboxes, classes=read_class_names(), show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.GFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
