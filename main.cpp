/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2022/9/20 9:56
 * @Version: 1.0
**/
#include "src/common.h"
#include "src/detection.h"
#include "src/pose.h"

cv::Mat drawSkeleton(cv::Mat& ori_img, std::vector<cv::Point>& points, int rec_x, int rec_y){

    for(auto& point : points){
        point.x = point.x + rec_x;
        point.y = point.y + rec_y;
    }

    for(auto & point : points){
        cv::circle(ori_img,
                   cv::Point(point.x,point.y),
                   4, cv::Scalar(0, 0, 255), -1);
    }
    cv::Point hip;
    hip.x = (points[8].x + points[9].x) / 2;
    hip.y = (points[8].y + points[9].y) / 2;
    std::vector<std::vector<int>> joints = {{0,1}, {2,3}, {2,4}, {4,6}, {3,5},
                                            {5,7}, {8,9}, {8,10}, {10,12},
                                            {9,11}, {11,13}};
    for(auto & joint : joints){
        cv::line(ori_img,
                 cv::Point(points[joint[0]].x, points[joint[0]].y),
                 cv::Point(points[joint[1]].x, points[joint[1]].y),
                 cv::Scalar(0, 0, 225), 2);
    }
    cv::circle(ori_img, cv::Point(hip.x, hip.y),
               4, cv::Scalar(0,0,255), -1);
    cv::line(ori_img, cv::Point(points[1].x, points[1].y), hip,
             cv::Scalar(0, 0, 225), 2);
    return ori_img;
}

int main(){

    std::string root_dir = "/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/";
    std::string nanodet_model_file = root_dir + "./model/nanodetplus_person.tflite";
    std::string mobilev2_model_file = root_dir + "./model/mobilenetv2_224_224_heatmap.tflite";
    std::string coco_name = root_dir + "./model/coco.names";
    std::string img_file = root_dir + "data/girls.jpg";

    cv::Mat ori_img = cv::imread(img_file);

    // nanodet人体检测调用
    detector::NanoDetPlus detect(nanodet_model_file, coco_name, 288, 512, 0.3, 0.4);
    // mobilenetv2人体姿态估计
    pose::Pose2d pose2D(mobilev2_model_file);
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
        drawSkeleton(ori_img, rec_joints, xmin, ymin);

        // 画出人体目标框和对应的人体姿态关键点
        rectangle(ori_img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1);
        string label = format("%.2f", boxes[i].score);
        vector<string> class_names = detect.get_class_names();
        label = class_names[boxes[i].label] + ":" + label;
        putText(ori_img, label, Point(xmin, ymin - 5),
                FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 0, 255), 2);
    }
    cv::imwrite("/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/data/detetor.png", ori_img);
    return 0;
}







