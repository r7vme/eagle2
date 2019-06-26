#include <chrono>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <TrtNet.h>
#include <YoloLayer.h>

#include "utils.hpp"


using namespace std;
using namespace Eigen;


namespace perception
{

tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn)
{
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph in the current session
  status = sess->Create(graph_def);
  if (status != tensorflow::Status::OK()) return status;

  return tensorflow::Status::OK();
}

vector<float> prepare_image(cv::Mat& img)
{
    int h=YOLO_H; int w=YOLO_W; int c=YOLO_C;

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, CV_INTER_CUBIC);

    cv::Mat cropped(h, w, CV_8UC3, 127);
    cv::Rect rect((w-scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width, scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (YOLO_C == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<cv::Mat> input_channels(c);
    split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void do_nms(vector<Yolo::Detection>& detections,int classes ,float nmsThresh)
{
    vector<vector<Yolo::Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Yolo::Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Yolo::Detection& left,const Yolo::Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);
}


vector<YoloBbox> postprocess_image(cv::Mat& img,vector<Yolo::Detection>& detections,int classes)
{
    int h=YOLO_H; int w=YOLO_W;

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    //nms
    float nmsThresh = YOLO_NMS_THRESH;
    if(nmsThresh > 0)
        do_nms(detections,classes,nmsThresh);

    vector<YoloBbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        YoloBbox bbox =
        { 
            item.classId,   //classId
            max(int((b[0]-b[2]/2.)*width),0), //left
            min(int((b[0]+b[2]/2.)*width),width), //right
            max(int((b[1]-b[3]/2.)*height),0), //top
            min(int((b[1]+b[3]/2.)*height),height), //bot
            item.prob       //score
        };
        boxes.push_back(bbox);
    }

    return boxes;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

//cv::Mat1b get_top_down_occupancy_array(const tensorflow::Tensor tr, const cv::Mat& H)
//{
//    auto tr_mapped = tr.tensor<int, 3>();
//
//    cv::Mat1b img(ENET_H, ENET_W, 100);
//    for( int i = 0; i < ENET_H; ++i) {
//      for( int j = 0; j < ENET_W; ++j ) {
//        int label = tr_mapped(0, i, j);
//        if (label==0)
//          img(i,j) = 0;
//      }
//    }
//    cv::resize(img, img, cv::Size(CAM_W, CAM_H), 0, 0, CV_INTER_LINEAR);
//    cv::Mat1b top;
//    cv::warpPerspective(img, top, H, cv::Size(TOP_W, TOP_H), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
//    return top;
//}


Points2D points3D_to_2D(const Points3D &pts3d,
                        const Vector3f &center,
                        const Matrix<float,3,3> &rot_M,
                        const Matrix<float,3,4> &P)
{
  Points2D pts;
  for (int i=0;i<pts3d.rows();++i)
  {
    Vector3f p3; Vector4f p4;
    p3 = pts3d.row(i).transpose();
    p3 = rot_M*p3 + center;
    p4 << p3,1.;
    p3 = P*p4;
    pts(i,0) = p3(0)/p3(2);
    pts(i,1) = p3(1)/p3(2);
  }
  return pts;
}

float compute_error(const Points2D &pts, const Vector4f &box_2D)
{
  Vector4f box_2D_new;
  box_2D_new << pts.col(0).minCoeff(),
                pts.col(1).minCoeff(),
                pts.col(0).maxCoeff(),
                pts.col(1).maxCoeff();
  float error = (box_2D_new-box_2D).cwiseAbs().sum();
  return error;
}

Points3D init_points3D(float h, float w, float l)
{
  Matrix<float,8,3> P3D;
  int id = 0;
  int ar[2]= {1,-1};
  for (auto i: ar)
  {
    for (auto j: ar)
    {
      for (auto k: ar)
      {
        P3D(id,0) = w/2.0*i;
        P3D(id,1) = h/2.0*k;
        P3D(id,2) = l/2.0*j*i;
        id+=1;
      }
    }
  }
  return P3D;
}

Vector3f compute_center(const Points3D &pts3d,
                        const Matrix<float,3,3> &rot_M,
                        const Matrix<float,3,4> &P,
                        const Vector4f &box_2D,
                        const int constants_id)
{
  float fx=P(0,0);
  float fy=P(1,1);
  float u0=P(0,2);
  float v0=P(1,2);
  MatrixXf A(4,3);
  A << fx, 0., u0-box_2D(0),
       fx, 0., u0-box_2D(2),
       0., fy, v0-box_2D(1),
       0., fy, v0-box_2D(3);

  Vector3f center;
  Vector3f result;
  float err_min = 1e10;
  for(auto &constraint : CONSTRAINTS[constants_id])
  {
    MatrixXf b(4,1);
    for (int i=0; i<4; ++i)
    {
      Vector3f p3;
      p3 = pts3d.row(constraint[i]);
      p3 = rot_M*p3;
      b(i) = box_2D(i)*P(2,3) - A.row(i).dot(p3) - P(i/2,3);
    }
    center = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
    Points2D pts=points3D_to_2D(pts3d, center, rot_M, P);
    float err = compute_error(pts, box_2D);
    if ((err<err_min) and (center(2)>0))
    {
      result=center;
      err_min=err;
    }
  }
  return result;
}

bool compute_3D_box(Bbox3D &bx, const Matrix<float,3,4> &P)
{
  float ray = bx.theta_ray;
  float yaw = bx.yaw;
  // sanity check
  if (!(ray>0. && ray<M_PI))
    return false;
  if (!(yaw>-2*M_PI && yaw<2*M_PI))
    return false;

  Matrix<float,3,3> rot_M;
  rot_M << cos(yaw),0.,sin(yaw),
           0.,      1.,      0.,
           sin(yaw),0.,cos(yaw);

  float norm_yaw = (yaw>0) ? yaw : yaw+2*M_PI;

  float deg0 = 0.;
  float deg89 = 89.*(M_PI/180.);
  float deg90 = 90.*(M_PI/180.);
  float deg179 = 179.*(M_PI/180.);
  float deg269 = 269.*(M_PI/180.);
  int constants_id;
  if (ray < deg90)
  {
    if (norm_yaw>deg0 && norm_yaw<deg89)
      constants_id=0;
    else if (norm_yaw>deg89 && norm_yaw<deg179)
      constants_id=1;
    else if (norm_yaw>deg179 && norm_yaw<deg269)
      constants_id=2;
    else
      constants_id=3;
  } else {
    if (norm_yaw>deg0 && norm_yaw<deg89)
      constants_id=4;
    else if (norm_yaw>deg89 && norm_yaw<deg179)
      constants_id=5;
    else if (norm_yaw>deg179 && norm_yaw<deg269)
      constants_id=6;
    else
      constants_id=7;
  }
  Points3D points3D = init_points3D(bx.h,bx.w,bx.l);
  Vector4f box_2D;
  box_2D << bx.xmin, bx.ymin, bx.xmax, bx.ymax;
  Vector3f center = compute_center(points3D, rot_M, P, box_2D, constants_id);
  bx.pts2d = points3D_to_2D(points3D, center, rot_M, P);
  return true;
}

void draw_3D_box(cv::Mat &img, const Points2D &pts)
{
  for (int i=0;i<4;++i)
  {
    cv::line(img,
             cv::Point(pts(2*i,0),pts(2*i,1)),
             cv::Point(pts(2*i+1,0),pts(2*i+1,1)),
             cv::Scalar(0,255,0),2,8);
  }
  cv::line(img,
           cv::Point(pts(0,0),pts(0,1)),
           cv::Point(pts(7,0),pts(7,1)),
           cv::Scalar(0,0,255),2,8);
  cv::line(img,
           cv::Point(pts(1,0),pts(1,1)),
           cv::Point(pts(6,0),pts(6,1)),
           cv::Scalar(0,0,255),2,8);
  for (int i=0;i<8;++i)
  {
    cv::line(img,
             cv::Point(pts(i,0),pts(i,1)),
             cv::Point(pts(fmod(i+2,8),0),pts(fmod(i+2,8),1)),
             cv::Scalar(0,255,0),2,8);
  }
}
} // namespace perception
