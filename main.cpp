/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2022/9/20 9:56
 * @Version: 1.0
**/
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define LOG(x) std::cerr

cv::Mat preprocessing(std::string img_file){

    int INPUT_SIZE = 224;
    std::vector<float> mean_vec = {0.485, 0.456, 0.406};
    std::vector<float> stddev_vec = {0.229, 0.226, 0.225};
    cv::Mat raw_image = cv::imread(img_file); // BGR format

    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    image.convertTo(image, CV_32FC3);

    cv::Mat outImage;
    outImage.create(image.rows, image.cols, CV_32FC3);
    for(int i=0; i<image.rows; ++i){
        for(int j=0; j<image.cols; ++j){
            outImage.at<cv::Vec3f>(i,j)[0] =
                    (image.at<cv::Vec3f>(i,j)[0]/255.0 - mean_vec[0]) / stddev_vec[0];
            outImage.at<cv::Vec3f>(i,j)[1] =
                    (image.at<cv::Vec3f>(i,j)[1]/255.0 - mean_vec[1]) / stddev_vec[1];
            outImage.at<cv::Vec3f>(i,j)[2] =
                    (image.at<cv::Vec3f>(i,j)[2]/255.0 - mean_vec[2]) / stddev_vec[2];
        }
    }
    return outImage;
}

void convertHWC2CHW(cv::Mat& img, float* model_input){
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

void heatmapToJoints(cv::Mat& ori_img, float* model_output, std::vector<cv::Point>& points){
    // heatmaps to joints == [1, 14, 28, 28]->[14, 2]
    points.clear();
    auto* feature_map = new float[28*28];
    for(int j=0; j<14; ++j){
        int index = 0;
        for(int i=0; i<28*28; ++i){
            feature_map[index] = (float)model_output[i+j*28*28];
            ++index;
        }
        cv::Mat feature_map_mat(28, 28, CV_32FC1, feature_map);
        double countMinVal = 0, countMaxVal = 0;
        cv::Point minPoint, maxPoint;
        cv::minMaxLoc(feature_map_mat,
                      &countMinVal, &countMaxVal,
                      &minPoint, &maxPoint);

        auto px = (float)maxPoint.x;
        auto py = (float)maxPoint.y;
        float x = px / 28 * ori_img.rows;
        float y = py / 28 * ori_img.cols;
        int xx = (int)x;
        int yy = (int)y;

        points.emplace_back(cv::Point(xx, yy));
    }
}

cv::Mat drawSkeleton(cv::Mat& ori_img, std::vector<cv::Point>& points){
    for(int i=0; i<points.size(); ++i){
        cv::circle(ori_img,
                   cv::Point(points[i].x,points[i].y),
                   10, cv::Scalar(0, 0, 255), -1);
    }
    cv::Point hip;
    hip.x = (points[8].x + points[9].x) / 2;
    hip.y = (points[8].y + points[9].y) / 2;
    std::vector<std::vector<int>> joints = {{0,1}, {2,3}, {2,4}, {4,6}, {3,5},
                                            {5,7}, {8,9}, {8,10}, {10,12},
                                            {9,11}, {11,13}};
    for(int i=0; i<joints.size(); ++i){
        cv::line(ori_img,
                 cv::Point(points[joints[i][0]].x, points[joints[i][0]].y),
                 cv::Point(points[joints[i][1]].x, points[joints[i][1]].y),
                 cv::Scalar(0, 0, 225), 3);
    }
    cv::circle(ori_img, cv::Point(hip.x, hip.y),
               10, cv::Scalar(0,0,255), -1);
    cv::line(ori_img, cv::Point(points[1].x, points[1].y), hip,
             cv::Scalar(0, 0, 225), 3);
    return ori_img;
}

int main(){

    std::string root_dir = "/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/";
    std::string model_file = root_dir + "./model/mobilenetv2_224_224_quant_heatmap.tflite";
    std::string img_file = root_dir + "./dancing.png";

    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    // 加载模型
    model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if(!model){
        LOG(FATAL) << "Failed to mmap model " << model_file << "\n";
    }

    // 将模型中tensor映射写入解释器对象中
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if(!interpreter){
        LOG(FATAL) << "Failed to construct interpreter ";
        exit(-1);
    }
    // 加载所有的tensor
    if(interpreter->AllocateTensors() != kTfLiteOk){
        LOG(FATAL) << "Failed to allocate tensors !";
    }
    // 多线程推理
    interpreter->SetNumThreads(1);

    // 定义输入对象
    int input =  interpreter->inputs()[0];
    LOG(INFO) << "input: " << input << "\n";
    // 获取输入tensor的维度
    TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
    LOG(INFO) << "input_dims.size: " << input_dims->size << "\n";
    LOG(INFO) << "input_dims[0]: " << input_dims->data[0] << "\n";
    LOG(INFO) << "input_dims[1]: " << input_dims->data[1] << "\n";
    LOG(INFO) << "input_dims[2]: " << input_dims->data[2] << "\n";
    LOG(INFO) << "input_dims[3]: " << input_dims->data[3] << "\n";

    cv::Mat ori_img = cv::imread(img_file);
    cv::Mat input_img = preprocessing(img_file); //[224, 224, 3]
    // 输入模型图片大小[1, 3, 224, 224]
    auto* model_input = interpreter->typed_tensor<float>(input);
    convertHWC2CHW(input_img, model_input);

    // 进行推理
    interpreter->Invoke();

    // 获取结果
    int output = interpreter->outputs()[0];
    LOG(INFO) << "output: " << output << "\n";
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    LOG(INFO) << "output_dims->size: " << output_dims->size << "\n";
    LOG(INFO) << "output_dims[0]: " << output_dims->data[0] << "\n";
    LOG(INFO) << "output_dims[1]: " << output_dims->data[1] << "\n";
    LOG(INFO) << "output_dims[2]: " << output_dims->data[2] << "\n";
    LOG(INFO) << "output_dims[3]: " << output_dims->data[3] << "\n";

    // 模型输出尺寸大小[1, 14, 28, 28]
    auto model_output = interpreter->typed_output_tensor<float>(0);

    std::vector<cv::Point> points;
    heatmapToJoints(ori_img, model_output, points);
    ori_img = drawSkeleton(ori_img, points);

    cv::imwrite("/mnt/e/WorkSpace/CPlusPlus/2d-pose-estimation/points.png", ori_img);
    LOG(INFO) << "tflite example finished" ;
    return 0;
}

