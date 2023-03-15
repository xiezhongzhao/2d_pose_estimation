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
        Pose2d(string& model_file, int height, int width);
        vector<cv::Point> getJoints(cv::Mat& ori_img);

    private:
        cv::Mat preprocessing(cv::Mat& ori_image);
        void convertHWC2CHW(cv::Mat& img_norm, float* model_input);
        void heatmapToJoints(cv::Mat& ori_img, float* model_output, std::vector<cv::Point>& points);
        cv::Mat drawSkeleton(cv::Mat& ori_img, std::vector<cv::Point>& points);

        string model_file;
        region reg{0, 0, 0, 0};
        int height;
        int width;
        const int heatmap_size = 28;
    };

    class Pose3d{
        // ...
    };
}

#endif //INC_2D_POSE_ESTIMATION_POSE_H


















