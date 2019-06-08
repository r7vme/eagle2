#pragma once
#include <opencv2/opencv.hpp>
#include <TrtNet.h>
#include <YoloLayer.h>


using namespace std;

namespace perception
{
    const int YOLO_C = 3;
    const int YOLO_W = 416;
    const int YOLO_H = 416;
    const int YOLO_NUM_CLS = 80;
    const float YOLO_NMS_THRESH = 0.45;

    struct YoloBbox
    {
        int classId;
        int left;
        int right;
        int top;
        int bot;
        float score;
    };

    vector<float> prepare_image(cv::Mat& img);
    void do_nms(vector<Yolo::Detection>& detections, int classes, float nmsThresh);
    vector<YoloBbox> postprocess_image(cv::Mat& img, vector<Yolo::Detection>& detections, int classes);
    vector<string> split(const string& str, char delim);
} // namespace perception
