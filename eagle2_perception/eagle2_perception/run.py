#! /usr/bin/env python
import time
import cv2
import numpy as np
import tensorflow as tf
import yaml

import utils_yolo as utils
import utils_box3d


def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)
    K = np.array(cfg["K"])
    D = np.array(cfg["D"])
    # projection matrix (mono) = intrinsics + 4th column with zeros
    P = np.zeros((3,4))
    P[:,:-1] = K
    assert K.shape == (3, 3)
    assert D.shape == (1, 5)
    assert P.shape == (3, 4)

    cam_id = cfg["cam_id"]
    cam_res = cfg["cam_res"]
    cam_fps = cfg["cam_fps"]
    dims_avg = np.array(cfg["voc_dims_avg"])
    do_viz = cfg["do_viz"]

    fx = K[0][0]
    u0 = K[0][2]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    yolo_pb = cfg["yolo_pb"]
    yolo_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                     "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    yolo_num_classes = 80
    yolo_input_size = 416
    yolo_graph = tf.Graph()
    yolo_tensors = utils.read_pb_return_tensors(yolo_graph, yolo_pb, yolo_elements)
    yolo_sess = tf.Session(graph=yolo_graph, config=config)

    b3d_pb = cfg["box3d_pb"]
    b3d_elements = ["input_1:0", "dimension/LeakyRelu:0",
                    "orientation/l2_normalize:0", "confidence/Softmax:0"]
    b3d_graph = tf.Graph()
    b3d_tensors = utils.read_pb_return_tensors(b3d_graph, b3d_pb, b3d_elements)
    b3d_sess = tf.Session(graph=b3d_graph, config=config)

    cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res[1])
    cam.set(cv2.CAP_PROP_FPS, cam_fps)
    # only supported from 4.1.0
    #cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = cv2.imread('kitti.png')
    while True:
        #ret, frame = cam.read()
        #if not ret:
        #    continue
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [yolo_input_size, yolo_input_size])
        image_data = image_data[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = yolo_sess.run(
            [yolo_tensors[1], yolo_tensors[2], yolo_tensors[3]],
            feed_dict={yolo_tensors[0]: image_data}
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + yolo_num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, yolo_input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        if do_viz:
            image = utils.draw_bbox(frame, bboxes)

        # 3D box
        bboxes = utils_box3d.filter_only_voc_class(bboxes)
        # For the sake of performance limit number of object estimations
        # TODO: check which objects are more important
        bboxes = bboxes[:8]
        patches = np.zeros((len(bboxes), 224, 224, 3), dtype=np.float32)
        for i, b in enumerate(bboxes):
            # prepare patch
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            patch = frame[ymin:ymax, xmin:xmax]
            patch = patch.astype(np.float32, copy=False)
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]]) # caffe norm
            patches[i] = patch

        # process patch
        predictions = b3d_sess.run(
            [b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]],
            feed_dict={b3d_tensors[0]: patches}
        )

        for i, b in enumerate(bboxes):
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            box_2D = np.asarray([xmin, ymin, xmax, ymax],dtype=np.float)
            cls = int(bboxes[i][5])
            pred = [[predictions[0][i]], [predictions[1][i]], [predictions[2][i]]]
            dims = dims_avg[cls] + pred[0][0]
            yaw, theta_ray = utils_box3d.compute_yaw(pred, xmin, xmax, fx, u0)
            points2D = utils_box3d.gen_3D_box(yaw, theta_ray, dims, P, box_2D)
            if do_viz:
                utils_box3d.draw_3D_box(image, points2D)

        if do_viz:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", image)
            cv2.imwrite("result.png", image)
            if cv2.waitKey(0) & 0xFF == ord('q'): break


if __name__ == "__main__":
    run()
