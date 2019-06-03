#! /usr/bin/env python
import time
import cv2
import numpy as np
import tensorflow as tf
import yaml

import utils


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

    K = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
                  [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    fx = K[0][0]
    fy = K[1][1]
    u0 = K[0][2]
    v0 = K[1][2]
    # Projeciton matrix
    P = np.hstack((K, [[0], [0], [0]]))

    # top down reprojection matrix (use calibrate.py)
    H = np.array([[ 1.38592897e-03,  8.66205605e-03, -2.34207693e+00],
                  [ 0.00000000e+00,  1.73241121e-02, -3.99454207e+00],
                  [ 0.00000000e+00,  3.46482242e-05, -5.98908415e-03]])

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

    enet_pb = cfg['enet_pb']
    enet_elements = ["imgs_ph:0", "early_drop_prob_ph:0", "late_drop_prob_ph:0",
                     "fullconv/Relu:0"]
    enet_graph = tf.Graph()
    enet_tensors = utils.read_pb_return_tensors(enet_graph, enet_pb, enet_elements)
    enet_sess = tf.Session(graph=enet_graph, config=config)

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
        image_data = utils.image_preporcess(frame.copy(), [yolo_input_size, yolo_input_size])
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
            viz_img = utils.draw_bbox(frame.copy(), bboxes)
            viz_top = np.zeros((500,500,3), np.uint8)

        # 3D box
        bboxes = utils.filter_only_voc_class(bboxes)
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
            patch = patch - utils.norm_caffe # caffe norm
            patches[i] = patch

        # process patch
        b3d_preds = b3d_sess.run(
            [b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]],
            feed_dict={b3d_tensors[0]: patches}
        )

        for i, b in enumerate(bboxes):
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            box_2D = np.asarray([xmin, ymin, xmax, ymax],dtype=np.float)
            cls = int(bboxes[i][5])
            pred = [[b3d_preds[0][i]], [b3d_preds[1][i]], [b3d_preds[2][i]]]
            dims = dims_avg[cls] + pred[0][0]
            yaw, theta_ray = utils.compute_yaw(pred, xmin, xmax, fx, u0)
            points2D, points3D = utils.gen_3D_box(yaw, theta_ray, dims, P, box_2D)
            if do_viz:
                utils.draw_3D_box(viz_img, points2D)
                p1x=int(250-points3D[0][1]*10)
                p1y=int(500-points3D[0][0]*10)
                p2x=int(250-points3D[2][1]*10)
                p2y=int(500-points3D[2][0]*10)
                p3x=int(250-points3D[4][1]*10)
                p3y=int(500-points3D[4][0]*10)
                p4x=int(250-points3D[6][1]*10)
                p4y=int(500-points3D[6][0]*10)
                cv2.line(viz_top,(p1x, p1y),(p2x, p2y),(0,255,0),1)
                cv2.line(viz_top,(p2x, p2y),(p3x, p3y),(0,255,0),1)
                cv2.line(viz_top,(p3x, p3y),(p4x, p4y),(0,255,0),1)
                cv2.line(viz_top,(p4x, p4y),(p1x, p1y),(0,255,0),1)
                cv2.line(viz_top,(250, 0),(250, 500),(255,0,0),1)

        enet_in = frame.astype(np.float32, copy=True)
        enet_in = cv2.resize(enet_in, (1024, 512))
        enet_in = enet_in - utils.norm_city
        enet_in = np.expand_dims(enet_in, axis=0)
        enet_pred = enet_sess.run(enet_tensors[3],
                                  feed_dict={
                                      enet_tensors[0]: enet_in,
                                      enet_tensors[1]: 0.0,
                                      enet_tensors[2]: 0.0
                                  })
        enet_pred = np.argmax(enet_pred, axis=3)

        if do_viz:
            viz_enet = enet_pred[0]
            viz_enet = utils.label_img_to_color(viz_enet)
            viz_enet += utils.norm_city
            viz_enet = viz_enet.astype(np.uint8, copy=False)
            viz_enet = cv2.resize(viz_enet, (frame_size[1], frame_size[0]))

            viz_img = cv2.addWeighted(viz_img, 0.5, viz_enet, 0.5, 0)
            cv2.imwrite("result.png", viz_img)
            cv2.imwrite("top.png", viz_top)

            #viz_warp = cv2.warpPerspective(frame.copy(),H,(500, 500))
            #viz_top = cv2.addWeighted(viz_top, 0.5, viz_warp, 0.5, 0)

            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("top", cv2.WINDOW_NORMAL)
            cv2.imshow("result", viz_img)
            cv2.imshow("top", viz_top)
            if cv2.waitKey(0) & 0xFF == ord('q'): break


if __name__ == "__main__":
    run()
