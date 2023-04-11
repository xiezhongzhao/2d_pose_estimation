/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:12
 * @Version: 1.0
**/

#ifndef INC_3D_POSE_ESTIMATION_INFERENCE_H
#define INC_3D_POSE_ESTIMATION_INFERENCE_H

#ifdef __ANDROID__
#define INTERFACE_DECLSPEC
#else
#define WIN32_LEAN_AND_MEAN             // 从 Windows 头文件中排除极少使用的内容
// Windows 头文件
//#include <windows.h>
#define INTERFACE_DECLSPEC __declspec(dllexport)
#endif

#include "common.h"

class TfliteInterface {
public:
    TfliteInterface();

    // 析构函数
    ~TfliteInterface();

    void infer(string& model_file, float* input, cv::Mat& img, float* result);

private:
    string model_file;
    bool allow_fp16 = false;
    bool gl_backend = false;
    bool hexagon_delegate = false;
    bool xnnpack_delegate = false;
    // nnapi, gl, xnnpack, hexagon
    std::string delegate_type = "cl";
    int number_of_threads = 1;

    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resovler;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

#endif //INC_3D_POSE_ESTIMATION_INFERENCE_H















