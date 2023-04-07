/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:12
 * @Version: 1.0
**/
#include <pose_estimation/tflite_inference.h>

#ifdef __ANDROID__
#ifdef BUILD_TFLITE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
#ifdef BUILD_TFLITE_HEXAGON
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#else
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif
#endif

TfliteInterface::TfliteInterface() = default;

TfliteInterface::~TfliteInterface() = default;


void TfliteInterface::infer(string& model_file, float* input, cv::Mat& img, float* result){
    // 加载模型
    model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if(!model){
        LOG(FATAL) << "Failed to mmap model " << model_file << "\n";
    }
    // 将模型中tensor映射写入解释器对象中
    tflite::InterpreterBuilder(*model, resovler)(&interpreter);
    if(!interpreter){
        LOG(FATAL) << "Failed to construct interpreter " << "\n";
    }
    // 加载所有tensor
    if(interpreter->AllocateTensors() != kTfLiteOk){
        LOG(FATAL) << "Failed to allocate tensors !" << "\n";
    }
    // 多线程推理
    interpreter->SetNumThreads(number_of_threads);
    // 定义输入对象
    int input_index =  interpreter->inputs()[0];
//    LOG(INFO) << "input_index: " << input_index << "\n";
    // 获取输入tensor的维度
    TfLiteIntArray* input_dims = interpreter->tensor(input_index)->dims;
//    LOG(INFO) << "input_dims.size: " << input_dims->size << "\n";
//    LOG(INFO) << "input_dims[0]: " << input_dims->data[0] << "\n";
//    LOG(INFO) << "input_dims[1]: " << input_dims->data[1] << "\n";
//    LOG(INFO) << "input_dims[2]: " << input_dims->data[2] << "\n";
//    LOG(INFO) << "input_dims[3]: " << input_dims->data[3] << "\n";
    TfLiteType input_type = interpreter->tensor(input_index)->type;
    if(input_type == kTfLiteFloat32){
        auto* model_input = interpreter->typed_tensor<float>(input_index);
        memcpy(model_input, input, img.rows*img.cols*img.channels()*sizeof(float));
        free(input);
    }else{
        auto* model_input = interpreter->typed_tensor<unsigned char>(input_index);
        memcpy(model_input, img.data, img.rows*img.cols*img.channels()*sizeof(unsigned char));
        free(input);
    }

    // 进行推理
    interpreter->Invoke();

    // 获取结果
    int output_index = interpreter->outputs()[0];
//    LOG(INFO) << "output_index: " << output_index << "\n";
    TfLiteIntArray* output_dims = interpreter->tensor(output_index)->dims;
//    LOG(INFO) << "output_dims->size: " << output_dims->size << "\n";
//    LOG(INFO) << "output_dims[0]: " << output_dims->data[0] << "\n";
//    LOG(INFO) << "output_dims[1]: " << output_dims->data[1] << "\n";
//    LOG(INFO) << "output_dims[2]: " << output_dims->data[2] << "\n";
//    LOG(INFO) << "output_dims[3]: " << output_dims->data[3] << "\n";
    TfLiteTensor* output_tensor = interpreter->tensor(output_index);
    if(output_tensor->type == kTfLiteUInt8){
        auto zero_point = output_tensor->params.zero_point;
        auto scale = output_tensor->params.scale;
        for(size_t j=0; j<output_tensor->bytes; ++j){
            result[j] = (float)(output_tensor->data.uint8[j] - zero_point) * scale;
        }
    }else{
        auto model_output = interpreter->typed_tensor<float>(output_index);
        memcpy(result,
               model_output,
               output_dims->data[0]*output_dims->data[1]*output_dims->data[2]*output_dims->data[3]*sizeof(float));
    }
}





















