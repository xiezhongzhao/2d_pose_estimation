/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:09
 * @Version: 1.0
**/

#ifndef INC_2D_POSE_ESTIMATION_POSE_H
#define INC_2D_POSE_ESTIMATION_POSE_H

#include "common.h"

namespace pose{

    typedef struct Region{
        int left;
        int top;
        int width;
        int height;
    }region;

    class Pose2d{
    public:
        Pose2d();
        explicit Pose2d(string& model_file);
        ~Pose2d();
        static cv::Mat preprocessing(cv::Mat& ori_image, int height, int width);
        void convertHWC2CHW(cv::Mat& img, float* model_input);
        void heatmapToJoints(cv::Mat& ori_img, float* model_output, std::vector<cv::Point>& points);
        cv::Mat drawSkeleton(cv::Mat& ori_img, std::vector<cv::Point>& points);
        vector<cv::Point> getJoints(cv::Mat& ori_img);
    private:
        string model_file;
    };

    class Pose3d{
        // ...
    };
}

#endif //INC_2D_POSE_ESTIMATION_POSE_H


















