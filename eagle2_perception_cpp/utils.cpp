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

Points3D init_points3D(const std::array<float, 3> &dims)
{
  Points3D points3D = MatrixXf::Zero(8, 3);
  return points3D;
//    cnt = 0 
//    for i in [1, -1]:
//        for j in [1, -1]:
//            for k in [1, -1]:
//                points3D[cnt] = dims[[1, 0, 2]].T / 2.0 * [i, k, j * i]
//                cnt += 1
//    return points3D
}

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

// - image with 0, 100 driving area
// - resize to orig size
// - warpperspective
cv::Mat1b get_top_down_occupancy_array(const tensorflow::Tensor tr, const cv::Mat& H)
{
    auto tr_mapped = tr.tensor<int, 3>();

    cv::Mat1b img(ENET_H, ENET_W, 100);
    for( int i = 0; i < ENET_H; ++i) {
      for( int j = 0; j < ENET_W; ++j ) {
        int label = tr_mapped(0, i, j);
        if (label==0)
          img(i,j) = 0;
      }
    }
    cv::resize(img, img, cv::Size(CAM_W, CAM_H), 0, 0, CV_INTER_LINEAR);
    cv::Mat1b top;
    cv::warpPerspective(img, top, H, cv::Size(TOP_W, TOP_H), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
    return top;
}

void label_image_to_color(const tensorflow::Tensor tr, cv::Mat &img)
{
    CV_Assert(img.rows==ENET_H && img.cols==ENET_W);

    auto tr_mapped = tr.tensor<int, 3>();

    cv::Mat_<cv::Vec3b> _I = img;
    for( int i = 0; i < ENET_H; ++i) {
      for( int j = 0; j < ENET_W; ++j ) {
        int label = tr_mapped(0, i, j);
        _I(i,j) = ENET_LABEL_TO_COLOR[label];
      }
    }
    img = _I;
}

//void compute_center()
//{

//def compute_center(points3D,rot_M,cam_to_img,box_2D, inds):
//    fx = cam_to_img[0][0]
//    fy = cam_to_img[1][1]
//    u0 = cam_to_img[0][2]
//    v0 = cam_to_img[1][2]
//    W = np.array([[fx, 0, float(u0 - box_2D[0])],
//                  [fx, 0, float(u0 - box_2D[2])],
//                  [0, fy, float(v0 - box_2D[1])],
//                  [0, fy, float(v0 - box_2D[3])]])
//    U, Sigma, VT = np.linalg.svd(W)

//}

} // namespace perception
