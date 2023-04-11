/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/4/10 15:50
 * @Version: 1.0
**/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pose_estimation/nanodet.h"
#include "pose_estimation/pose2d.h"
#include "pose_estimation/euro_filter.h"

namespace py = pybind11;

void detectImg(py::array_t<uint8_t>& input_img, string root_path){

    Info info;
    info.root_path = root_path;
    info.nanodet_path = Info::addRootPath(info.root_path, "model/nanodetplus_person.tflite");
    info.class_path = Info::addRootPath(info.root_path, "model/coco.names");
    info.pose_path = Info::addRootPath(info.root_path,"model/mobilenetv2_224_224_heatmap.tflite");
    info.img_in_path = Info::addRootPath(info.root_path,"data/family.jpg");
    info.img_out_path = Info::addRootPath(info.root_path,"data/family_out.jpg");
    info.video_in_path = Info::addRootPath(info.root_path,"data/black_man.mp4");
    info.video_out_path = Info::addRootPath(info.root_path,"data/black_man_out.avi");

    string nanodet_model_file = info.nanodet_path;
    string mobilev2_model_file = info.pose_path;
    string coco_name = info.class_path;
    string image_in_path = info.img_in_path;
    string image_out_path = info.img_out_path;

//    cv::Mat ori_img = cv::imread(image_in_path);
    auto rows = input_img.shape(0);
    auto cols = input_img.shape(1);
    auto type = CV_8UC3;
    cv::Mat ori_img(rows, cols, type, (unsigned char*)input_img.data());

    // nanodet人体检测调用
    detector::NanoDetPlus detect(nanodet_model_file, coco_name, 288, 512, 0.3, 0.4);
    // mobilenetv2人体姿态估计
    pose::Pose2d pose2D(mobilev2_model_file, 224, 224);
    // 所有人体区域
    vector<detector::BoxInfo> boxes = detect.object_bboxes(ori_img);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        int xmin = (int)boxes[i].x1;
        int ymin = (int)boxes[i].y1;
        int xmax = (int)boxes[i].x2;
        int ymax = (int)boxes[i].y2;

        // 裁剪原始图像获得人体区域
        cv::Rect rect(xmin, ymin, xmax-xmin, ymax-ymin);
        cv::Mat img_roi = ori_img(rect);
//        cv::imwrite("/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/img_roi.png", img_roi);

        // 人体区域输入姿态模型
        vector<cv::Point> rec_joints = pose2D.getJoints(img_roi);

        // 获得人体姿态关键点，并且投影会原始图像
        pose2D.drawSkeleton(ori_img, rec_joints, xmin, ymin);

        // 画出人体目标框和对应的人体姿态关键点
        rectangle(ori_img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1);
        string label = format("%.2f", boxes[i].score);
        vector<string> class_names = detect.get_class_names();
        label = class_names[boxes[i].label] + ":" + label;
        putText(ori_img, label, Point(xmin, ymin - 5),
                FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 0, 255), 2);
    }
    cout << "Finished !!!" << endl;
}

//int main(){
//    cv::Mat img;
//    string root_path = "/mnt/e/WorkSpace/CPlusPlus/2d_pose_estimation/";
//    detectImg(img, root_path);
//    return 0;
//}

PYBIND11_MODULE(inference, m){
    m.doc() = "human pose estimation";
    m.def("detectImg", &detectImg, "a function for human pose estimation");
}












