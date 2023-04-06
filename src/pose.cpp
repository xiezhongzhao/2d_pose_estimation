/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:09
 * @Version: 1.0
**/
#include "pose.h"
#include "inference.h"

namespace pose{

    Pose2d::Pose2d(std::string &model_file, int height, int width) {
        this->model_file = model_file;
        this->height = height;
        this->width = width;
    }

    cv::Mat Pose2d::preprocessing(cv::Mat& raw_image){
        int src_h = raw_image.rows, src_w = raw_image.cols;
        int dst_h = this->height, dst_w = this->width;

        float h = (float)dst_w * (static_cast<float>(src_h)/static_cast<float>(src_w));
        float w = (float)dst_h * (static_cast<float>(src_w)/static_cast<float>(src_h));
        cv::Mat image_dst;
        if(h <= (float)dst_h){
            cv::resize(raw_image, image_dst, Size(dst_w, static_cast<int>(h)));
        }else{
            cv::resize(raw_image, image_dst, Size(static_cast<int>(w), dst_h));
        }
        int h_ = image_dst.rows, w_ = image_dst.cols;
        int top = (dst_h - h_) / 2;
        int bottom = (dst_h - h_ + 1) / 2;
        int left = (dst_w - w_) / 2;
        int right = (dst_w - w_ + 1) / 2;
        cv::copyMakeBorder(image_dst, image_dst, top, bottom, left, right, cv::BORDER_CONSTANT);
        // 保持输入图像原始比例，将图像大小resize为224*224
        // 将缩放后的原始图像在224*224图像的位置保存在私有标量area里

        this->reg.left = left, this->reg.top = top;
        this->reg.width = dst_w - left - right;
        this->reg.height = dst_h - top - bottom;

        std::vector<float> mean_vec = {0.485, 0.456, 0.406};
        std::vector<float> stddev_vec = {0.229, 0.226, 0.225};

        image_dst.convertTo(image_dst, CV_32FC3);

        cv::Mat outImage;
        outImage.create(image_dst.rows, image_dst.cols, CV_32FC3);
        for(int i=0; i<image_dst.rows; ++i){
            for(int j=0; j<image_dst.cols; ++j){
                outImage.at<cv::Vec3f>(i,j)[0] =
                        (image_dst.at<cv::Vec3f>(i,j)[0]/255.0 - mean_vec[0]) / stddev_vec[0];
                outImage.at<cv::Vec3f>(i,j)[1] =
                        (image_dst.at<cv::Vec3f>(i,j)[1]/255.0 - mean_vec[1]) / stddev_vec[1];
                outImage.at<cv::Vec3f>(i,j)[2] =
                        (image_dst.at<cv::Vec3f>(i,j)[2]/255.0 - mean_vec[2]) / stddev_vec[2];
            }
        }
        return outImage;
    }

    void Pose2d::convertHWC2CHW(cv::Mat& img, float* model_input){
        // 输入模型图片大小[1, 224, 224, 3] -> [1, 3, 224, 224]
        auto* flatten = new float[img.rows * img.cols * img.channels()];
        int k = 0;
        for(int c=0; c<3; ++c){
            for(int i=0; i<img.rows; ++i){
                for(int j=0; j<img.cols; ++j){
                    if(c == 0){
                        flatten[k] = (float)img.at<cv::Vec3f>(i,j)[0];
                    }
                    else if(c == 1){
                        flatten[k] = (float)img.at<cv::Vec3f>(i,j)[1];
                    }
                    else{
                        flatten[k] = (float)img.at<cv::Vec3f>(i,j)[2];
                    }
                    ++k;
                }
            }
        }
        for(int i=0; i < img.cols * img.rows * img.channels(); ++i){
            model_input[i] = flatten[i];
        }
    }

    void Pose2d::heatmapToJoints(cv::Mat& ori_img, float* model_output, std::vector<cv::Point>& points){
        // heatmaps to joints == [1, 14, 28, 28]->[14, 2]
        points.clear();
        int pixels_heatmap = this->heatmap_size * this->heatmap_size;
        auto* feature_map = new float[pixels_heatmap];
        for(int j=0; j<14; ++j){
            int index = 0;
            for(int i=0; i<pixels_heatmap; ++i){
                feature_map[index] = (float)model_output[i+j*pixels_heatmap];
                ++index;
            }
            cv::Mat feature_map_mat(this->heatmap_size, this->heatmap_size, CV_32FC1, feature_map);
            double countMinVal = 0, countMaxVal = 0;
            cv::Point minPoint, maxPoint;
            cv::minMaxLoc(feature_map_mat,
                          &countMinVal, &countMaxVal,
                          &minPoint, &maxPoint);

            auto px = (float)maxPoint.x;
            auto py = (float)maxPoint.y;
            float x = px / this->heatmap_size * this->width;
            float y = py / this->heatmap_size * this->height;
            int xx = (int)x;
            int yy = (int)y;

            int start_x = this->reg.left, start_y = this->reg.top;
            int reg_width = this->reg.width, reg_height = this->reg.height;
            float w_times = (float)ori_img.cols / static_cast<float>(reg_width);
            float h_times = (float)ori_img.rows / static_cast<float>(reg_height);
            xx = static_cast<int>((float)(xx - start_x) * w_times);
            yy = static_cast<int>((float)(yy - start_y) * h_times);

            points.emplace_back(cv::Point(xx, yy));
        }
    }

    cv::Mat Pose2d::drawSkeleton(cv::Mat& ori_img, std::vector<cv::Point>& points, int rec_x, int rec_y){

        for(auto& point : points){
            point.x = point.x + rec_x;
            point.y = point.y + rec_y;
        }

        for(auto & point : points){
            cv::circle(ori_img,
                       cv::Point(point.x,point.y),
                       6, cv::Scalar(0, 0, 255), -1);
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
                     cv::Scalar(0, 0, 225), 3);
        }
        cv::circle(ori_img, cv::Point(hip.x, hip.y),
                   6, cv::Scalar(0,0,255), -1);
        cv::line(ori_img, cv::Point(points[1].x, points[1].y), hip,
                 cv::Scalar(0, 0, 225), 3);
        return ori_img;
    }

    vector<cv::Point> Pose2d::getJoints(cv::Mat& img) {

        cv::Mat input_img = preprocessing(img); //[224, 224, 3]
        // 输入模型图片大小[224, 224, 3]->[1, 3, 224, 224]
        float* model_input = new float[input_img.rows * input_img.cols * input_img.channels()];
        convertHWC2CHW(input_img, model_input);

        TfliteInterface interface;
        float* model_output = new float[14*28*28];
        interface.infer(model_file, model_input, input_img, model_output);

        std::vector<cv::Point> points;
        heatmapToJoints(img, model_output, points);

//        drawSkeleton(img, points);
//        cv::imwrite("/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/points.png", img);
//        LOG(INFO) << "the pose tflite example finished" << "\n";
        return points;
    }

};























