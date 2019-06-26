#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <TrtNet.h>
#include <yaml-cpp/yaml.h>
#include <YoloLayer.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/OccupancyGrid.h>

#include "bonnet.hpp"
#include "utils.hpp"


using namespace std;
using namespace bonnet;
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
  string cam_id               = config["cam_id"].as<string>();
  string cam_frame_id         = config["cam_frame_id"].as<string>();
  bool do_viz                 = config["do_viz"].as<bool>();
  string bonnet_engine        = config["bonnet_engine"].as<string>();
  string yolo_engine          = config["yolo_engine"].as<string>();
  string box3d_pb             = config["box3d_pb"].as<string>();
  vector<vector<float>> H_vec = config["H"].as<vector<vector<float>>>();
  vector<vector<float>> K_vec = config["K"].as<vector<vector<float>>>();
  vector<vector<float>> D_vec = config["D"].as<vector<vector<float>>>();

  // prepare CV matrices
  cv::Mat H = toMat(H_vec);
  cv::Mat K = toMat(K_vec);
  cv::Mat D = toMat(D_vec);
  float fx = K.at<float>(0,0);
  float u0 = K.at<float>(0,2);
  // projection (extrinsic)
  // assume world frame aligned with camera frame
  Matrix<float,3,4> P;
  P << K_vec[0][0],K_vec[0][1],K_vec[0][2],0.,
       K_vec[1][0],K_vec[1][1],K_vec[1][2],0.,
       K_vec[2][0],K_vec[2][1],K_vec[2][2],0.;

  // yolo (TensorRT)
  unique_ptr<Tn::trtNet> yolo;
  // TODO: it can fail silently, check if initialized.
  yolo.reset(new Tn::trtNet(yolo_engine));

  // bonnet (TensorRT)
  unique_ptr<Bonnet> bonnet;
  bonnet.reset(new Bonnet(bonnet_engine));
  if (!bonnet->initialized)
  {
    // TOOD: add log msg
    ROS_ERROR("Failed to initialize bonnet");
    ros::shutdown();
    return -1;
  }

  // box3d (TensorFlow)
  tensorflow::SessionOptions tf_options;
  tf_options.config.mutable_gpu_options()->set_allow_growth(true);
  tensorflow::Session *b3d_sess;
  TF_CHECK_OK(tensorflow::NewSession(tf_options, &b3d_sess));
  TF_CHECK_OK(LoadModel(b3d_sess, box3d_pb));
  vector<string> b3d_tensors{"input_1","dimension/LeakyRelu",
                             "orientation/l2_normalize","confidence/Softmax"};

  // v4l camera capture. cam_id also can be just a path to a video
  cv::VideoCapture cap(cam_id);
  if (!cap.isOpened())
     return -1;
  cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
  cap.set(cv::CAP_PROP_FOCUS, 0);
  cap.set(cv::CAP_PROP_FPS, CAM_FPS);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, CAM_W);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_H);
  // only from 4.2
  // cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

  cv::Mat frame;
  cv::Mat frame_raw;
  cv::Mat frame_viz;
  cv::Mat bonnet_output;
  cv_bridge::CvImage ros_image;
  nav_msgs::OccupancyGrid map_msg;
  map_msg.data.resize(TOP2_W*TOP2_H);
  while(ros::ok())
  {
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
    auto bboxes = postprocess_image(frame, result, YOLO_NUM_CLS);

    // 3D part that estimates 3d bounding boxes:
    // - estimate only cars, because we have compute limited capacity
    // - estimate pedestrians 3d box w/o NN, just assume yaw=0;
    vector<YoloBbox> bboxes_cars;
    vector<YoloBbox> bboxes_peds;
    for(const auto& bbox: bboxes)
    {
      YoloBbox b = bbox;
      // if car
      if ((b.classId==2) and (bboxes_cars.size() < B3D_MAX_OBJECTS))
      {
        b.classId=0;
        bboxes_cars.push_back(b);
      }
      // if pedestrian
      if (b.classId==0)
      {
        b.classId=3;
        bboxes_peds.push_back(b);
      }
    }

    int batch_size = bboxes_cars.size();
    // prepare inputs
    tensorflow::Tensor b3d_input(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size, B3D_H, B3D_W, B3D_C}));
    auto b3d_input_mapped = b3d_input.tensor<float, 4>();

    {
      for (int b = 0; b < batch_size; ++b) {
        YoloBbox bbox = bboxes_cars[b];
        cv::Rect patchRect(
          cv::Point(bbox.left,bbox.top),
          cv::Point(bbox.right,bbox.bot)
        );
        cv::Mat patch;
        cv::resize(frame(patchRect), patch, cv::Size(B3D_W, B3D_H), 0, 0, CV_INTER_CUBIC);
        patch.convertTo(patch, CV_32FC3);
        subtract(patch,NORM_CAFFE,patch);
        const float * source_data = (float*) patch.data;
        // copying the data into the corresponding tensor
        for (int y = 0; y < B3D_H; ++y) {
          const float* source_row = source_data + (y * B3D_W * B3D_C);
          for (int x = 0; x < B3D_W; ++x) {
            const float* source_pixel = source_row + (x * B3D_C);
            for (int c = 0; c < B3D_C; ++c) {
              const float* source_value = source_pixel + c;
              b3d_input_mapped(b, y, x, c) = *source_value;
            }
          }
        }
      }
    }
    auto t_start = chrono::high_resolution_clock::now();
    vector<tensorflow::Tensor> b3d_output;
    // running the loaded graph
    tensorflow::Status b3d_status  = b3d_sess->Run(
      {{b3d_tensors[0], b3d_input}},
      {b3d_tensors[1], b3d_tensors[2], b3d_tensors[3]},
      {},
      &b3d_output
    );
    if (!b3d_status.ok())
      continue;
    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken is " << total << " ms." << endl;

    vector<Bbox3D> bboxes3d;
    auto dimensions=b3d_output[0].tensor<float, 2>();
    auto orientation=b3d_output[1].tensor<float, 3>();
    auto confidence=b3d_output[2].tensor<float, 2>();
    for (int b = 0; b < batch_size; ++b)
    {
      // find bin ID with max confidence
      Eigen::Matrix<float, B3D_BIN_NUM, 1> conf;
      for (int i = 0; i < B3D_BIN_NUM; ++i)
        conf[i]=confidence(b, i);

      int max_a; conf.maxCoeff(&max_a);
      float cos_v = orientation(b, max_a, 0);
      float sin_v = orientation(b, max_a, 1);

      // compute yaw
      int xmin = bboxes_cars[b].left;
      int ymin = bboxes_cars[b].top;
      int xmax = bboxes_cars[b].right;
      int ymax = bboxes_cars[b].bot;
      float theta_ray = atan2(fx, ((xmin+xmax)/2.)-u0);
      float angle_offset = (sin_v > 0.) ? acos(cos_v) : -acos(cos_v);
      float wedge = (2.*M_PI)/B3D_BIN_NUM;
      float theta_loc = angle_offset + max_a * wedge;
      float theta = theta_loc + theta_ray;
      float yaw = fmod((M_PI/2)-theta, 2*M_PI); // make yaw in betwen [-2Pi,2Pi]

      // fill the values
      Bbox3D bbox3d;
      bbox3d.h=dimensions(b, 0) + DIMS_AVG[0][0];
      bbox3d.w=dimensions(b, 1) + DIMS_AVG[0][1];
      bbox3d.l=dimensions(b, 2) + DIMS_AVG[0][2];
      bbox3d.yaw=yaw;
      bbox3d.theta_ray=theta_ray;
      bbox3d.xmin=xmin;
      bbox3d.ymin=ymin;
      bbox3d.xmax=xmax;
      bbox3d.ymax=ymax;
      // hmm, just skip?
      if (!compute_3D_box(bbox3d, P))
        continue;
      bboxes3d.push_back(bbox3d);
    }

    // bonnet
    bonnet->doInference(frame, bonnet_output);
    // 1. H - Homography matrix was computed with calibrate.py for !!!512x154!!! --> 500x500
    // 2. TOP1_RES - Resolution px/cm was computed using known size objects
    // 3. TOP2_RES is desired resolution (All TOP2 values are derived from TOP1)
    // 4. CAM_TO_BOTTOM_EDGE camera original position from bottom edge of image
    //    was computed by using known height and assuming zero pitch and yaw
    cv::Mat1b top;
    cv::warpPerspective(bonnet_output, top, H,
                        cv::Size(TOP1_W, TOP1_H),
                        CV_INTER_LINEAR,
                        cv::BORDER_CONSTANT, 255);
    cv::resize(top, top, cv::Size(TOP2_W, TOP2_H), 0, 0);
    cv::flip(top, top, 0);
    map_msg.header.stamp = frame_stamp;
    map_msg.header.frame_id = cam_frame_id;
    map_msg.info.map_load_time = frame_stamp;
    map_msg.info.resolution = TOP2_RES;
    map_msg.info.width = TOP2_W;
    map_msg.info.height = TOP2_H;
    map_msg.info.origin.position.x = CAMERA_ORIGIN_X;
    map_msg.info.origin.position.y = CAMERA_ORIGIN_Y;
    map_msg.info.origin.position.z = 0.0;
    map_msg.info.origin.orientation.x = 0.0;
    map_msg.info.origin.orientation.y = 0.0;
    map_msg.info.origin.orientation.z = -0.707;
    map_msg.info.origin.orientation.w = 0.707;

    uchar* p = top.data;
    for( int i = 0; i < TOP2_W*TOP2_H; ++i)
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

    map_pub.publish(map_msg);

   // cv::namedWindow("top",CV_WINDOW_AUTOSIZE);
   // cv::imshow("top",top);
   // cv::waitKey(1);

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
      //cv::namedWindow("window",CV_WINDOW_AUTOSIZE);
      //cv::imshow("window", frame_viz);
      //cv::namedWindow("top",CV_WINDOW_AUTOSIZE);
      //cv::imshow("top", top_down);
      //cv::waitKey(1);
    }
  }
  return 0;
}