#include <chrono>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <TrtNet.h>
#include <yaml-cpp/yaml.h>
#include <YoloLayer.h>

#include "utils.hpp"


using namespace std;
using namespace perception;

int main()
{
  YAML::Node config = YAML::LoadFile("config.yaml");

  // get config
  string cam_id              = config["cam_id"].as<string>();
  string cam_frame_id        = config["cam_frame_id"].as<string>();
  bool do_viz                = config["do_viz"].as<bool>();
  vector<vector<float>> H_vec = config["H"].as<vector<vector<float>>>();
  vector<vector<float>> K_vec = config["K"].as<vector<vector<float>>>();
  cv::Mat H = toMat(H_vec);
  cv::Mat K = toMat(K_vec);
  float fx = K.at<float>(0,0);
  float u0 = K.at<float>(0,2);

  string yolo_engine = config["yolo_engine"].as<string>();
  string box3d_pb = config["box3d_pb"].as<string>();
  string enet_pb = config["enet_pb"].as<string>();

  // yolo (TensorRT)
  unique_ptr<Tn::trtNet> net;
  // TODO: it can fail silently, check if initialized.
  net.reset(new Tn::trtNet(yolo_engine));

  // tensorflow session options
  tensorflow::SessionOptions tf_options;
  tf_options.config.mutable_gpu_options()->set_allow_growth(true);

  // box3d (TensorFlow)
  tensorflow::Session *b3d_sess;
  TF_CHECK_OK(tensorflow::NewSession(tf_options, &b3d_sess));
  TF_CHECK_OK(LoadModel(b3d_sess, box3d_pb));
  vector<string> b3d_tensors{"input_1","dimension/LeakyRelu",
                             "orientation/l2_normalize","confidence/Softmax"};

  // enet (TensorFlow)
  tensorflow::Session *enet_sess;
  TF_CHECK_OK(tensorflow::NewSession(tf_options, &enet_sess));
  TF_CHECK_OK(LoadModel(enet_sess, enet_pb));
  vector<string> enet_tensors{"imgs_ph","early_drop_prob_ph",
                              "late_drop_prob_ph","argmax_1"};

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
  while(true)
  {
    if (!cap.read(frame))
      continue;

    auto t_start = std::chrono::high_resolution_clock::now();
    vector<float> input_data = prepare_image(frame);

    int output_count = net->getOutputSize()/sizeof(float);
    unique_ptr<float[]> output_data(new float[output_count]);

    net->doInference(input_data.data(), output_data.get(), 1);

    //Get Output
    auto output = output_data.get();
    //first detect count
    int detCount = output[0];
    //later detect result
    vector<Yolo::Detection> result;
    result.resize(detCount);
    memcpy(result.data(), &output[1], detCount*sizeof(Yolo::Detection));
    auto bboxes = postprocess_image(frame, result, YOLO_NUM_CLS);

    //draw on image
    for(const auto& bbox: bboxes)
    {
      cv::rectangle(
        frame,
        cv::Point(bbox.left,bbox.top),
        cv::Point(bbox.right,bbox.bot),
        cv::Scalar(0,0,255), 3, 8, 0
      );
    }

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

    vector<Bbox3D> bboxes3d;
    auto dimensions=b3d_output[0].tensor<float, 2>();
    auto orientation=b3d_output[1].tensor<float, 3>();
    auto confidence=b3d_output[2].tensor<float, 2>();
    for (int b = 0; b < batch_size; ++b) {
      int xmin = bboxes_cars[b].left;
      int xmax = bboxes_cars[b].right;

      Bbox3D bbox3d;
      bbox3d.h=dimensions(b, 0) + DIMS_AVG[0][0];
      bbox3d.w=dimensions(b, 1) + DIMS_AVG[0][1];
      bbox3d.l=dimensions(b, 2) + DIMS_AVG[0][2];
      // find bin ID with max confidence
      Eigen::Matrix<float, B3D_BIN_NUM, 1> conf;
      for (int i = 0; i < B3D_BIN_NUM; ++i)
        conf[i]=confidence(b, i);

      int max_a; conf.maxCoeff(&max_a);
      float cos_v = orientation(b, max_a, 0);
      float sin_v = orientation(b, max_a, 1);

      // compute yaw
      float theta_ray = atan2(fx, ((xmin+xmax)/2.)-u0);
      float angle_offset = (sin_v > 0.) ? acos(cos_v) : -acos(cos_v);
      float wedge = (2.*M_PI)/B3D_BIN_NUM;
      float theta_loc = angle_offset + max_a * wedge;
      float theta = theta_loc + theta_ray;
      float yaw = (M_PI/2)-theta;
    }

//        # 3D box
//        for i, b in enumerate(bboxes):
//            xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
//            box_2D = np.asarray([xmin, ymin, xmax, ymax], dtype=np.float)
//            cls = int(bboxes[i][5])
//            pred = [[b3d_preds[0][i]], [b3d_preds[1][i]], [b3d_preds[2][i]]]
//            dims = dims_avg[cls] + pred[0][0]
//            yaw, theta_ray = utils.compute_yaw(pred, xmin, xmax, fx, u0)
//            pts = utils.gen_3D_box(yaw, theta_ray, dims, P, box_2D)
//            if do_viz:
//                utils.draw_3D_box(viz_img, pts)

    // enet
    // prepare inputs
    tensorflow::Tensor enet_input(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, ENET_H, ENET_W, ENET_C}));
    auto enet_input_mapped = enet_input.tensor<float, 4>();

    cv::Mat enet_in;
    cv::resize(frame, enet_in, cv::Size(ENET_W, ENET_H), 0, 0, CV_INTER_CUBIC);
    enet_in.convertTo(enet_in, CV_32FC3);
    subtract(enet_in,NORM_CITY,enet_in);

    {
      const float * source_data = (float*) enet_in.data;
      // copying the data into the corresponding tensor
      for (int y = 0; y < ENET_H; ++y) {
        const float* source_row = source_data + (y * ENET_W * ENET_C);
        for (int x = 0; x < ENET_W; ++x) {
          const float* source_pixel = source_row + (x * ENET_C);
          for (int c = 0; c < ENET_C; ++c) {
            const float* source_value = source_pixel + c;
            enet_input_mapped(0, y, x, c) = *source_value;
          }
        }
      }
    }
    tensorflow::Tensor enet_input2(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    enet_input2.scalar<float>()() = 0.0;
    tensorflow::Tensor enet_input3(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    enet_input3.scalar<float>()() = 0.0;

    vector<tensorflow::Tensor> enet_output;
    tensorflow::Status enet_status  = enet_sess->Run(
      {
        {enet_tensors[0], enet_input},
        {enet_tensors[1], enet_input2},
        {enet_tensors[2], enet_input3}
      },
      {enet_tensors[3]},
      {},
      &enet_output
    );
    if (!enet_status.ok())
      continue;

    //cv::Mat img(ENET_H, ENET_W, CV_8UC3);
    //label_image_to_color(enet_output[0], img);
    cv::Mat1b top_down = get_top_down_occupancy_array(enet_output[0], H);
    //cv::imwrite("1.jpg", top_down);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Time taken for inference is " << total << " ms." << std::endl;
  }
  return 0;
}
