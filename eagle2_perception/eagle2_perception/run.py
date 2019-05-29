#! /usr/bin/env python
import cv2
import numpy as np
import utils_yolo as utils
import utils_box3d
import time
import tensorflow as tf


def run():
    cam_id = 0
    cam_res = (800, 448)
    cam_fps = 10
    # projection matrix (mono) = intrinsics + 4th column with zeros
    P = np.array([[5.9564893980982720e+02, 0., 3.9092986149050319e+02, 0.],
                  [0., 5.9564893980982720e+02, 2.0740193446502806e+02, 0.],
                  [0., 0., 1., 0.]])
    D = np.array([[3.8650985712807895e-02,
                   -1.7020510263944139e-01,
                   0.,
                   0.,
                   1.2632002448706023e-01]])
    dims_avg = np.array([
        [1.52131309, 1.64441358, 3.85728004],
        [2.18560847, 1.91077601, 5.08042328],
        [3.07044968, 2.62877944, 11.17126338],
        [1.75562272, 0.67027992, 0.87397566],
        [1.28627907, 0.53976744, 0.96906977],
        [1.73456498, 0.58174006, 1.77485499],
        [3.56020305, 2.40172589, 18.60659898]])
    fx = P[0][0]
    u0 = P[0][2]

    yolo_pb = "./yolov3_coco.pb"
    yolo_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                     "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    yolo_num_classes = 80
    yolo_input_size = 416
    yolo_graph = tf.Graph()
    yolo_tensors = utils.read_pb_return_tensors(yolo_graph, yolo_pb, yolo_elements)
    yolo_sess = tf.Session(graph=yolo_graph)

    b3d_pb = "./box3d.pb"
    b3d_elements = ["input_1:0", "dimension/LeakyRelu:0",
                    "orientation/l2_normalize:0", "confidence/Softmax:0"]
    b3d_graph = tf.Graph()
    b3d_tensors = utils.read_pb_return_tensors(b3d_graph, b3d_pb, b3d_elements)
    b3d_sess = tf.Session(graph=b3d_graph)

    cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res[1])
    cam.set(cv2.CAP_PROP_FPS, cam_fps)

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
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
        # (xmin, ymin, xmax, ymax, score, class)
        # visualize
        #image = utils.draw_bbox(frame, bboxes)

        for b in bboxes:
            cls = utils_box3d.coco_to_voc_class(int(b[5]))
            if not cls:
                continue
            # prepare patch
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            patch = frame[ymin:ymax, xmin:xmax]
            patch = patch.astype(np.float32, copy=False)
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]]) # caffe norm
            patch = np.expand_dims(patch, 0)

            # process patch
            prediction = b3d_sess.run(
                [b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]],
                feed_dict={b3d_tensors[0]: patch}
            )

            dims = dims_avg[cls] + prediction[0][0]
            yaw = utils_box3d.compute_yaw(prediction, xmin, xmax, fx, u0)
            #points2D = utils_box3d.gen_3D_box(yaw, dims, P, b[:4])
            #utils_box3d.draw_3D_box(image, points2D)
        #cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        #cv2.imshow("result", image)
        #if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == "__main__":
    run()
