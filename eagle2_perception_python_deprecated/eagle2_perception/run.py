#! /usr/bin/env python
import time
import os
import cv2
import numpy as np
import tensorflow as tf
import yaml

import utils


def run():
    with open('config.yaml', 'r') as f:
        cfg = yaml.load(f)

    # camera parameters
    K = np.array(cfg["K"])
    D = np.array(cfg["D"])
    # projection matrix (mono) = intrinsics + 4th column with zeros
    P = np.zeros((3, 4))
    P[:, :-1] = K
    H = np.array(cfg["H"])
    assert K.shape == (3, 3)
    assert D.shape == (1, 5)
    assert P.shape == (3, 4)
    assert H.shape == (3, 3)
    fx = K[0][0]
    u0 = K[0][2]

    cam_id = cfg["cam_id"]
    cam_res = cfg["cam_res"]
    cam_fps = cfg["cam_fps"]
    cam_frame_id = cfg["cam_frame_id"]
    dims_avg = np.array(cfg["voc_dims_avg"])
    do_viz = cfg["do_viz"]
    images_path = cfg["images_path"]
    use_ros = cfg["use_ros"]
    top_down_width = cfg["top_down_width"]
    top_down_height = cfg["top_down_height"]
    top_down_resolution = cfg["top_down_resolution"]

    if use_ros:
        import rospy
        from cv_bridge import CvBridge
        from eagle2_msgs.msg import PerceptionStamped, Object
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import Image

    # camera
    if not images_path:
        cam = cv2.VideoCapture(cam_id)
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cam.set(cv2.CAP_PROP_FOCUS, 0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res[1])
        cam.set(cv2.CAP_PROP_FPS, cam_fps)
        # only supported from 4.1.0
        # cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        files = sorted(os.listdir(images_path))

        def img_gen_func():
            for f in files:
                if os.path.splitext(f)[-1] != ".png":
                    continue
                yield cv2.imread(os.path.join(images_path, f))
        img_gen = img_gen_func()

    # TENSORFLOW
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True

    # object detection network
    yolo_pb = cfg["yolo_pb"]
    yolo_elements = ["input/input_data:0", "pred_sbbox/concat_2:0",
                     "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    yolo_num_classes = 80
    yolo_input_size = 416
    yolo_graph = tf.Graph()
    yolo_tensors = utils.read_pb_return_tensors(yolo_graph, yolo_pb, yolo_elements)
    yolo_sess = tf.Session(graph=yolo_graph, config=config)

    # 3D bounding boxes estimation network
    b3d_pb = cfg["box3d_pb"]
    b3d_elements = ["input_1:0", "dimension/LeakyRelu:0",
                    "orientation/l2_normalize:0", "confidence/Softmax:0"]
    b3d_graph = tf.Graph()
    b3d_tensors = utils.read_pb_return_tensors(b3d_graph, b3d_pb, b3d_elements)
    b3d_sess = tf.Session(graph=b3d_graph, config=config)

    # segmentation network
    enet_pb = cfg['enet_pb']
    enet_elements = ["imgs_ph:0", "early_drop_prob_ph:0", "late_drop_prob_ph:0" ,"argmax_1:0"]
    enet_graph = tf.Graph()
    enet_tensors = utils.read_pb_return_tensors(enet_graph, enet_pb, enet_elements)
    enet_sess = tf.Session(graph=enet_graph, config=config)

    # ROS
    if use_ros:
        rospy.init_node('eagle2_perception')
        ros_cv_bridge = CvBridge()
        ros_pub_perception = rospy.Publisher('perception', PerceptionStamped)
        if do_viz:
            ros_pub_viz = rospy.Publisher('perception_viz', Image)

    while True:
        if use_ros and rospy.is_shutdown():
            break

        if use_ros:
            ros_msg = PerceptionStamped()
            ros_msg.header.frame_id = cam_frame_id
            ros_msg.header.stamp = rospy.Time.now()

        if not images_path:
            ret, frame = cam.read()
            if not ret:
                continue
        else:
            frame = next(img_gen)

        if do_viz:
            viz_img = frame.copy()


        # YOLO
        frame_size = frame.shape[:2]
        yolo_input = utils.image_preporcess(frame.copy(), [yolo_input_size, yolo_input_size])
        yolo_input = yolo_input[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = yolo_sess.run(
            [yolo_tensors[1], yolo_tensors[2], yolo_tensors[3]],
            feed_dict={yolo_tensors[0]: yolo_input}
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + yolo_num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + yolo_num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, yolo_input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        image = utils.draw_bbox(frame, bboxes)
        cv2.imwrite("result.png", image)


        # 3D box
        bboxes = utils.filter_only_voc_class(bboxes)
        # For the sake of performance limit number of object estimations
        # TODO: check which objects are more important
        bboxes = bboxes[:4]
        b3d_input = np.zeros((len(bboxes), 224, 224, 3), dtype=np.float32)
        for i, b in enumerate(bboxes):
            # prepare patch
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            patch = frame[ymin:ymax, xmin:xmax]
            patch = patch.astype(np.float32, copy=False)
            patch = cv2.resize(patch, (224, 224))
            patch = patch - utils.norm_caffe  # caffe norm
            b3d_input[i] = patch

        # process patch
        b3d_preds = b3d_sess.run(
            [b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]],
            feed_dict={b3d_tensors[0]: b3d_input}
        )

        for i, b in enumerate(bboxes):
            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            box_2D = np.asarray([xmin, ymin, xmax, ymax], dtype=np.float)
            cls = int(bboxes[i][5])
            pred = [[b3d_preds[0][i]], [b3d_preds[1][i]], [b3d_preds[2][i]]]
            dims = dims_avg[cls] + pred[0][0]
            t1 = time.time()
            yaw, theta_ray = utils.compute_yaw(pred, xmin, xmax, fx, u0)
            t2 = time.time()
            print("time %.2fms" % (1000*(t2-t1)))
            pts = utils.gen_3D_box(yaw, theta_ray, dims, P, box_2D)
            if do_viz:
                utils.draw_3D_box(viz_img, pts)
            if use_ros:
                obj = Object()
                obj.label = cls
                obj.height = dims[0]
                obj.width = dims[1]
                obj.length = dims[2]
                obj.yaw = yaw
                # 0,2,4,6 are bottom points
                obj.bbox = [pts[0][0], pts[0][1],
                            pts[2][0], pts[2][1],
                            pts[4][0], pts[4][1],
                            pts[6][0], pts[6][1]]
                ros_msg.data.objects.append(obj)

        # ENet
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
        #enet_pred = np.argmax(enet_pred, axis=3)
        if use_ros:
            grid = cv2.warpPerspective(enet_pred[0], H, (500, 500))
            with np.nditer(grid, op_flags=['readwrite']) as it:
                for x in it:
                    if x == 0:
                        x[...] = 0
                    else:
                        x[...] = 100
            og = OccupancyGrid()
            og.header.stamp = ros_msg.header.stamp
            og.header.frame_id = cam_frame_id
            og.info.resolution = top_down_resolution
            og.info.width = top_down_width
            og.info.height = top_down_height
            og.info.origin.position.x = 0.0
            og.info.origin.position.y = (top_down_width/2)*top_down_resolution
            og.info.origin.position.z = 0.0
            og.info.origin.orientation.x = 0.0
            og.info.origin.orientation.y = 0.0
            og.info.origin.orientation.z = 0.0
            og.info.origin.orientation.w = 1.0
            og.data = grid.reshape(-1).tolist()
            ros_msg.data.drivable_area = og
            ros_pub_perception.publish(ros_msg)

        if do_viz:
            viz_enet = enet_pred[0]
            viz_enet = utils.label_img_to_color(viz_enet)
            viz_enet += utils.norm_city
            viz_enet = viz_enet.astype(np.uint8, copy=False)
            viz_enet = cv2.resize(viz_enet, (frame_size[1], frame_size[0]))
            viz_img = cv2.addWeighted(viz_img, 0.5, viz_enet, 0.5, 0)

            if use_ros:
                viz_msg = ros_cv_bridge.cv2_to_imgmsg(viz_img, encoding="passthrough")
                viz_msg.header.stamp = ros_msg.header.stamp
                ros_pub_viz.publish(viz_msg)
            else:
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", viz_img)
                cv2.waitKey(10)


if __name__ == "__main__":
    run()
