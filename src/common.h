/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/3/13 10:00
 * @Version: 1.0
**/

#ifndef INC_2D_POSE_ESTIMATION_COMMON_H
#define INC_2D_POSE_ESTIMATION_COMMON_H

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cstdint>

#include <opencv2/opencv.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#define LOG(x) std::cerr

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::unique_ptr;
using std::max;
using std::min;
using std::ifstream;
using std::ios;
using std::strcmp;
using std::move;

using cv::Mat;
using cv::Point2d;
using cv::Point;
using cv::Scalar;
using cv::format;
using cv::FONT_HERSHEY_SIMPLEX;
using cv::Size;
using cv::INTER_AREA;
using cv::BORDER_CONSTANT;

#endif //INC_2D_POSE_ESTIMATION_COMMON_H











