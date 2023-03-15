/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:08
 * @Version: 1.0
**/

#ifndef INC_2D_POSE_ESTIMATION_DETECTION_H
#define INC_2D_POSE_ESTIMATION_DETECTION_H

#include "common.h"

namespace detector{

    typedef struct BoxInfo {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } BoxInfo;

    class NanoDetPlus {
    public:
        NanoDetPlus(string& model_path,
                    string& classesFile,
                    int height, int width,
                    float nms_threshold,
                    float objThreshold);

        void detect(Mat &cv_image);

        vector<string> get_class_names();
        vector<BoxInfo> object_bboxes(Mat& cv_image);

    private:
        string model_file;
        float score_threshold = 0.5;
        float nms_threshold = 0.5;
        std::vector<string> class_names;
        int num_class;

        Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);

        void normalize(Mat &srcimg);

        void softmax_(const float *x, float *y, int length);

        void generate_proposal(vector<BoxInfo> &generate_boxes, const float *preds);

        void nms(vector<BoxInfo> &input_boxes);

        const bool keep_ratio = true;
        int inpWidth;
        int inpHeight;
        const int reg_max = 7;
        const int num_stages = 4;
        const int stride[4] = {8, 16, 32, 64};

//    const float mean[3] = { 103.53, 116.28, 123.675 };
//    const float std[3] = { 57.375, 57.12, 58.395 };
        const float mean[3] = {0, 0, 0};
        const float std[3] = {255, 255, 255};

        bool isQuant = true; //默认为量化模型
    };
}
#endif //INC_2D_POSE_ESTIMATION_DETECTION_H















