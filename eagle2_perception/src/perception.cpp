#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <TrtNet.h>
#include <yaml-cpp/yaml.h>
#include <YoloLayer.h>

#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/OccupancyGrid.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>

#include "bonnet.hpp"
#include "box3d.hpp"
#include "utils.hpp"


using namespace std;
using namespace perception;

int main(int argc, char **argv)
{
  // ROS
  ros::init(argc, argv, "eagle2_perception");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");

  string config_file;
  priv_nh.param<string>("config_file", config_file, "config.yaml");

  ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("image_rect_color", 1);
  ros::Publisher map_pub   = nh.advertise<nav_msgs::OccupancyGrid>("drivable_area", 1);
  ros::Publisher bbox_pub  = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("bboxes", 1);

  YAML::Node config;
  try
  {
    config = YAML::LoadFile(config_file);
  }
  catch (const YAML::BadFile&)
  {
    ROS_ERROR_STREAM("Failed to read config file at "<<config_file);
    ros::shutdown();
    return -1;
  }

  // get config
  // TODO: catch YAML expection
  string cam_id                = config["cam_id"].as<string>();
  string cam_frame_id          = config["cam_frame_id"].as<string>();
  int cam_width                = config["cam_width"].as<int>();
  int cam_height               = config["cam_height"].as<int>();
  float cam_height_from_ground = config["cam_height_from_ground"].as<float>();
  int cam_fps                  = config["cam_fps"].as<int>();
  int top_width                = config["top_width"].as<int>();
  int top_height               = config["top_height"].as<int>();
  float top_res                = config["top_res"].as<float>();
  float top2_res               = config["top2_res"].as<float>();
  bool do_viz                  = config["do_viz"].as<bool>();
  string bonnet_engine         = config["bonnet_engine"].as<string>();
  string yolo_engine           = config["yolo_engine"].as<string>();
  string box3d_engine          = config["box3d_engine"].as<string>();
  vector<vector<float>> H_vec  = config["H"].as<vector<vector<float>>>();
  vector<vector<float>> K_vec  = config["K"].as<vector<vector<float>>>();
  vector<vector<float>> D_vec  = config["D"].as<vector<vector<float>>>();

  // prepare CV matrices
  cv::Mat H = toMat(H_vec);
  cv::Mat K = toMat(K_vec);
  cv::Mat D = toMat(D_vec);
  float fx = K.at<float>(0,0);
  float fy = K.at<float>(1,1);
  float u0 = K.at<float>(0,2);
  float v0 = K.at<float>(1,2);
  // projection (extrinsic)
  // assume world frame aligned with camera frame
  Matrix<float,3,4> P;
  P << K_vec[0][0],K_vec[0][1],K_vec[0][2],0.,
       K_vec[1][0],K_vec[1][1],K_vec[1][2],0.,
       K_vec[2][0],K_vec[2][1],K_vec[2][2],0.;

  // occupancy grid size with desired resolution (10cm/px)
  int top2_width  = top_width * (top_res/top2_res);
  int top2_height = top_height * (top_res/top2_res);
  // camera origin from topdown in meters
  float cam_origin_x = cam_height_from_ground*(fy/v0);
  float cam_origin_y = (top2_width/2)*top2_res;
  float bonnet_scale = min(float(bonnet::W)/cam_width,float(bonnet::H)/cam_height);

  // yolo (TensorRT)
  unique_ptr<Tn::trtNet> yolo;
  // TODO: it can fail silently, check if initialized.
  yolo.reset(new Tn::trtNet(yolo_engine));

  // bonnet (TensorRT)
  unique_ptr<bonnet::Bonnet> bonnet_net;
  bonnet_net.reset(new bonnet::Bonnet(bonnet_engine));
  if (!bonnet_net->initialized)
  {
    // TOOD: add log msg
    ROS_ERROR("Failed to initialize bonnet");
    ros::shutdown();
    return -1;
  }

  // box3d (TensorRT)
  unique_ptr<box3d::Box3D> box3d_net;
  box3d_net.reset(new box3d::Box3D(box3d_engine));
  if (!box3d_net->initialized)
  {
    // TOOD: add log msg
    ROS_ERROR("Failed to initialize box3d");
    ros::shutdown();
    return -1;
  }

  // v4l camera capture. cam_id also can be just a path to a video
  cv::VideoCapture cap(cam_id);
  if (!cap.isOpened())
     return -1;
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);
  cap.set(cv::CAP_PROP_FPS, cam_fps);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
  // only from 4.2
  // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  // also setup viz stuff if enabled
  ros::Publisher viz_pub;
  if (do_viz)
  {
    viz_pub = nh.advertise<sensor_msgs::Image>("image_viz", 2);
  }

  cv::Mat frame;
  cv::Mat frame_raw;
  cv::Mat frame_viz;
  cv::Mat bonnet_output;
  cv_bridge::CvImage ros_image;
  nav_msgs::OccupancyGrid map_msg;
  map_msg.data.resize(top2_width*top2_height);
  while(ros::ok())
  {
    auto t_start = chrono::high_resolution_clock::now();
    if (!cap.read(frame_raw))
    {
      continue;
    }

    ros::Time frame_stamp=ros::Time::now();
    cv::undistort(frame_raw, frame, K, D);

    ros_image.header.stamp=frame_stamp;
    ros_image.header.frame_id=cam_frame_id;
    ros_image.encoding="bgr8";
    ros_image.image=frame;
    image_pub.publish(ros_image.toImageMsg());

    // YOLO
    vector<float> input_data = prepare_image(frame);
    int output_count = yolo->getOutputSize()/sizeof(float);
    unique_ptr<float[]> output_data(new float[output_count]);

    yolo->doInference(input_data.data(), output_data.get(), 1);

    //Get Output
    auto output = output_data.get();
    //first detect count
    int detCount = output[0];
    //later detect result
    vector<Yolo::Detection> result;
    result.resize(detCount);
    memcpy(result.data(), &output[1], detCount*sizeof(Yolo::Detection));
    auto bboxes_all = postprocess_image(frame, result, YOLO_NUM_CLS);

    // 3D part that estimates 3d bounding boxes:
    // - estimate only cars, because we have compute limited capacity
    // - estimate pedestrians 3d box w/o NN, just assume yaw=0;
    vector<YoloBbox> bboxes;
    vector<YoloBbox> bboxes_peds;
    for(const auto& bbox: bboxes_all)
    {
      YoloBbox b = bbox;
      // all except pedestrian
      if ((b.classId==2) or
          (b.classId==5) or
          (b.classId==7) or
          (b.classId==1) or
          (b.classId==6))
      {
        int coco_class = b.classId;
        b.classId=COCO_TO_VOC.at(coco_class);
        bboxes.push_back(b);
      // pedestrian
      } else if (b.classId==0)
      {
        int coco_class = b.classId;
        b.classId=COCO_TO_VOC.at(coco_class);
        bboxes_peds.push_back(b);
      }
    }

    // sort by distance from ego car
    sort(bboxes.begin(), bboxes.end(), &YoloBboxCompareDistance);

    int batch_size;
    if (bboxes.size()>box3d::MAX_BATCH_SIZE)
    {
      batch_size = box3d::MAX_BATCH_SIZE;
    } else {
      batch_size = bboxes.size();
    }

    vector<float> b3d_input_data(batch_size * box3d::INPUT_SIZE);
    vector<float> b3d_out_dims(batch_size * box3d::DIMS_SIZE);
    vector<float> b3d_out_ori(batch_size * box3d::ORI_SIZE);
    vector<float> b3d_out_conf(batch_size * box3d::CONF_SIZE);
    {
      auto buf = b3d_input_data.data();
      cv::Mat patch;
      vector<cv::Mat> b3d_input_channels(box3d::C);
      for (int b = 0; b < batch_size; ++b) {
        // prepare patch
        YoloBbox bbox = bboxes[b];
        cv::Rect patchRect(
          cv::Point(bbox.left,bbox.top),
          cv::Point(bbox.right,bbox.bot)
        );
        cv::resize(frame(patchRect), patch, cv::Size(box3d::W, box3d::H), 0, 0, CV_INTER_CUBIC);
        patch.convertTo(patch, CV_32FC3);
        subtract(patch,NORM_CAFFE,patch); // Caffe norm
        // split channels (HWC->CHW) and copy into buffer
        split(patch, b3d_input_channels);
        int channel_len = box3d::W * box3d::H;
        for (int i = 0; i < box3d::C; ++i) {
            memcpy(buf, b3d_input_channels[i].data, channel_len * sizeof(float));
            buf += channel_len;
        }
      }
    }
    if (batch_size > 0)
    {
      box3d_net->doInference(batch_size, b3d_input_data, b3d_out_dims,
                             b3d_out_ori, b3d_out_conf);
    }

    // estimate 3D bounding box based on NN outputs
    vector<Bbox3D> bboxes3d;
    for (int b = 0; b < batch_size; ++b)
    {
      // find bin ID with max confidence
      Eigen::Matrix<float, box3d::BIN_NUM, 1> conf;
      for (int i = 0; i < box3d::BIN_NUM; ++i)
        conf[i]=b3d_out_conf[b*box3d::CONF_SIZE+i];

      int max_a; conf.maxCoeff(&max_a);
      float cos_v = b3d_out_ori[b*box3d::ORI_SIZE + max_a*2];
      float sin_v = b3d_out_ori[b*box3d::ORI_SIZE + max_a*2 + 1];

      // compute yaw
      int xmin = bboxes[b].left;
      int ymin = bboxes[b].top;
      int xmax = bboxes[b].right;
      int ymax = bboxes[b].bot;
      float theta_ray = atan2(fx, ((xmin+xmax)/2.)-u0);
      float angle_offset = (sin_v > 0.) ? acos(cos_v) : -acos(cos_v);
      float wedge = (2.*M_PI)/box3d::BIN_NUM;
      float theta_loc = angle_offset + max_a * wedge;
      float theta = theta_loc + theta_ray;
      float yaw = fmod((M_PI/2)-theta, 2*M_PI); // make yaw in betwen [-2Pi,2Pi]

      // fill the values
      Bbox3D bbox3d;
      bbox3d.h=b3d_out_dims[b*box3d::DIMS_SIZE]     + DIMS_AVG[bboxes[b].classId][0];
      bbox3d.w=b3d_out_dims[b*box3d::DIMS_SIZE + 1] + DIMS_AVG[bboxes[b].classId][1];
      bbox3d.l=b3d_out_dims[b*box3d::DIMS_SIZE + 2] + DIMS_AVG[bboxes[b].classId][2];
      bbox3d.yaw=yaw;
      bbox3d.theta_ray=theta_ray;
      bbox3d.xmin=xmin;
      bbox3d.ymin=ymin;
      bbox3d.xmax=xmax;
      bbox3d.ymax=ymax;
      // hmm, just skip?
      if (!compute_3D_box(bbox3d, P))
        continue;

      // filter boxes with big estimatation error
      if (bbox3d.estimation_error > B3D_FILTER_ESTIMATION_ERROR)
        continue;

      // filter boxes vertical boxex near sides (this does not include people!!!)
      bool vertical = (abs(bbox3d.xmin-bbox3d.xmax)/abs(bbox3d.ymin-bbox3d.ymax) < 1.0);
      bool near_side = (((cam_width-bbox3d.xmax)<B3D_FILTER_SIDE_PX) or (bbox3d.xmin<B3D_FILTER_SIDE_PX));
      if ((near_side) and (vertical))
        continue;

      bboxes3d.push_back(bbox3d);
    }

    // estimtate 3D bounding box for pedestrians (w/o NN outputs)
    // for the sake of performance assumed yaw=0 and dimentions are average
    for(const auto& bbox: bboxes_peds)
    {
      Bbox3D bbox3d;
      bbox3d.h=DIMS_AVG[3][0];
      bbox3d.w=DIMS_AVG[3][1];
      bbox3d.l=DIMS_AVG[3][2];
      bbox3d.yaw=0.0;
      bbox3d.theta_ray=0.0;
      bbox3d.xmin=bbox.left;
      bbox3d.ymin=bbox.top;
      bbox3d.xmax=bbox.right;
      bbox3d.ymax=bbox.bot;
      // hmm, just skip?
      if (!compute_3D_box(bbox3d, P))
        continue;

      bboxes3d.push_back(bbox3d);
    }

    // bonnet
    bonnet_net->doInference(frame, bonnet_output);

    jsk_recognition_msgs::BoundingBoxArray bbox_arr_msg;
    bbox_arr_msg.header.stamp = frame_stamp;
    bbox_arr_msg.header.frame_id = cam_frame_id;

    // 1. H - Homography matrix was computed with calibrate.py for !!!512x154!!! --> 500x500
    // 2. top_res - Resolution px/cm was computed using known size objects
    // 3. top2_res is desired resolution (All top2 values are derived from top1)
    // 4. CAM_TO_BOTTOM_EDGE camera original position from bottom edge of image
    //    was computed by using known height and assuming zero pitch and yaw
    cv::Mat1b top;
    cv::warpPerspective(bonnet_output, top, H,
                        cv::Size(top_width, top_height),
                        CV_INTER_LINEAR,
                        cv::BORDER_CONSTANT, 255);
    cv::resize(top, top, cv::Size(top2_width, top2_height), 0, 0);
    cv::flip(top, top, 0);
    map_msg.header.stamp = frame_stamp;
    map_msg.header.frame_id = cam_frame_id;
    map_msg.info.map_load_time = frame_stamp;
    map_msg.info.resolution = top2_res;
    map_msg.info.width = top2_width;
    map_msg.info.height = top2_height;
    map_msg.info.origin.position.x = cam_origin_x;
    map_msg.info.origin.position.y = cam_origin_y;
    map_msg.info.origin.position.z = 0.0;
    map_msg.info.origin.orientation.x = 0.0;
    map_msg.info.origin.orientation.y = 0.0;
    map_msg.info.origin.orientation.z = -0.707;
    map_msg.info.origin.orientation.w = 0.707;
    uchar* p = top.data;
    for( int i = 0; i < top2_width*top2_height; ++i)
    {
        int v=p[i];
        if (v>100)
        {
          map_msg.data[i]=255;
        }
        else if (v>ROAD_THRESH)
        {
          map_msg.data[i]=0;
        }
        else
        {
          map_msg.data[i]=100;
        }
    }

    for(const auto& bbox3d: bboxes3d)
    {
      jsk_recognition_msgs::BoundingBox bbox_msg;
      bbox_msg.header.stamp = frame_stamp;
      bbox_msg.header.frame_id = cam_frame_id;
      bbox_msg.label = bbox3d.label;
      bbox_msg.dimensions.x = bbox3d.l;
      bbox_msg.dimensions.y = bbox3d.w;
      bbox_msg.dimensions.z = bbox3d.h;
      tf2::Quaternion q;
      q.setRPY(0,0,-bbox3d.yaw); // invert yaw
      bbox_msg.pose.orientation.x = q[0];
      bbox_msg.pose.orientation.y = q[1];
      bbox_msg.pose.orientation.z = q[2];
      bbox_msg.pose.orientation.w = q[3];
      tuple<int,int> coords = reproject_to_ground(bbox3d.proj_center_x*bonnet_scale,
                                                  bbox3d.proj_center_y*bonnet_scale, H);
      // x axis in ros is (img_height-y axis) in image
      // y axis in ros is -x axis in image
      int x = ((top_height - get<1>(coords)) * top_res) + cam_origin_x;
      int y = (-get<0>(coords) * top_res) + cam_origin_y;
      bbox_msg.pose.position.x = x;
      bbox_msg.pose.position.y = y;
      bbox_msg.pose.position.z = bbox3d.h/2;
      bbox_msg.value = 0;
      bbox_arr_msg.boxes.push_back(bbox_msg);
    }

    // finally publish map and bboxes
    if (ros::ok())
    {
      bbox_pub.publish(bbox_arr_msg);
      map_pub.publish(map_msg);
    }
    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();

    if (do_viz)
    {
      frame_viz = frame.clone();

      //draw YOLO boxes on image
      for(const auto& bbox: bboxes)
      {
        cv::rectangle(
          frame_viz,
          cv::Point(bbox.left,bbox.top),
          cv::Point(bbox.right,bbox.bot),
          cv::Scalar(0,0,255), 3, 8, 0
        );
      }

      for(const auto& bbox3d: bboxes3d)
      {
        draw_3D_box(frame_viz, bbox3d.pts2d);
      }

      cv::putText(frame_viz, string("FPS ") + to_string(1000/total), cv::Point(30,30),
        cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(0,0,255), 1, CV_AA);

      cv_bridge::CvImage ros_image_viz;
      ros_image_viz.header.stamp=frame_stamp;
      ros_image_viz.header.frame_id=cam_frame_id;
      ros_image_viz.encoding="bgr8";
      ros_image_viz.image=frame_viz;
      viz_pub.publish(ros_image_viz.toImageMsg());
    }
  }
  return 0;
}
