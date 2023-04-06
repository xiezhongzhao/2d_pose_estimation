/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2023/2/20 17:08
 * @Version: 1.0
**/

#include "detection.h"
#include "inference.h"

namespace detector{

    NanoDetPlus::NanoDetPlus(string& model_path,
                             string& classesFile,
                             int height, int width, float nms_threshold, float objThreshold)
    {
        std::ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line))
            this->class_names.push_back(line);
        this->num_class = class_names.size();
        this->nms_threshold = nms_threshold;
        this->score_threshold = objThreshold;

        this->inpHeight = height;
        this->inpWidth = width;

        this->model_file = model_path;

    }

    Mat NanoDetPlus::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
    {
        int srch = srcimg.rows, srcw = srcimg.cols;
        *newh = this->inpHeight;
        *neww = this->inpWidth;
        Mat dstimg;
        if (this->keep_ratio && srch != srcw) {
            float hw_scale = (float)srch / srcw;
            if (hw_scale > 1) {
                *newh = this->inpHeight;
                *neww = int(this->inpWidth / hw_scale);
                resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
                *left = int((this->inpWidth - *neww) * 0.5);
                copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
            }
            else {
                *newh = (int)this->inpHeight * hw_scale;
                *neww = this->inpWidth;
                resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
                *top = (int)(this->inpHeight - *newh) * 0.5;
                copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
            }
        }
        else {
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
        }
//        imshow("dstimg", dstimg);
        return dstimg;
    }

    void NanoDetPlus::normalize(Mat& img)
    {
        img.convertTo(img, CV_32FC3);
        int i = 0, j = 0;
        for (i = 0; i < img.rows; i++)
        {
            float* pdata = (float*)(img.data + i * img.step);
            for (j = 0; j < img.cols; j++)
            {
                pdata[0] = (pdata[0] - this->mean[0]) / this->std[0];
                pdata[1] = (pdata[1] - this->mean[1]) / this->std[1];
                pdata[2] = (pdata[2] - this->mean[2]) / this->std[2];
                pdata += 3;
            }
        }
    }

    void NanoDetPlus::softmax_(const float* x, float* y, int length)
    {
        float sum = 0;
        int i = 0;
        for (i = 0; i < length; i++)
        {
            y[i] = exp(x[i]);
            sum += y[i];
        }
        for (i = 0; i < length; i++)
        {
            y[i] /= sum;
        }
    }

    void NanoDetPlus::generate_proposal(vector<BoxInfo>& generate_boxes, const float* preds)
    {
        const int reg_1max = reg_max + 1;
        const int len = this->num_class + 4 * reg_1max;
        for (int n = 0; n < this->num_stages; n++)
        {
            const int stride_ = this->stride[n];
            const int num_grid_y = (int)ceil((float)this->inpHeight / stride_);
            const int num_grid_x = (int)ceil((float)this->inpWidth / stride_);
            ////cout << "num_grid_x=" << num_grid_x << ",num_grid_y=" << num_grid_y << endl;

            for (int i = 0; i < num_grid_y; i++)
            {
                for (int j = 0; j < num_grid_x; j++)
                {
                    int max_ind = 0;
                    float max_score = 0;
                    for (int k = 0; k < num_class; k++)
                    {
                        if (preds[k] > max_score)
                        {
                            max_score = preds[k];
                            max_ind = k;
                        }
                    }
                    if (max_score >= score_threshold)
                    {
                        const float* pbox = preds + this->num_class;
                        float dis_pred[4];
                        float* y = new float[reg_1max];
                        for (int k = 0; k < 4; k++)
                        {
                            softmax_(pbox + k * reg_1max, y, reg_1max);
                            float dis = 0.f;
                            for (int l = 0; l < reg_1max; l++)
                            {
                                dis += l * y[l];
                            }
                            dis_pred[k] = dis * stride_;
                        }
                        delete[] y;
                        /*float pb_cx = (j + 0.5f) * stride_ - 0.5;
                        float pb_cy = (i + 0.5f) * stride_ - 0.5;*/
                        float pb_cx = j * stride_;
                        float pb_cy = i * stride_;
                        float x0 = pb_cx - dis_pred[0];
                        float y0 = pb_cy - dis_pred[1];
                        float x1 = pb_cx + dis_pred[2];
                        float y1 = pb_cy + dis_pred[3];
                        generate_boxes.push_back(BoxInfo{ x0, y0, x1, y1, max_score, max_ind });
                    }
                    preds += len;
                }
            }
        }

    }

    void NanoDetPlus::nms(vector<BoxInfo>& input_boxes)
    {
        sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
        vector<float> vArea(input_boxes.size());
        for (int i = 0; i < int(input_boxes.size()); ++i)
        {
            vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                       * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
        }

        vector<bool> isSuppressed(input_boxes.size(), false);
        for (int i = 0; i < int(input_boxes.size()); ++i)
        {
            if (isSuppressed[i]) { continue; }
            for (int j = i + 1; j < int(input_boxes.size()); ++j)
            {
                if (isSuppressed[j]) { continue; }
                float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
                float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
                float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
                float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

                float w = (max)(float(0), xx2 - xx1 + 1);
                float h = (max)(float(0), yy2 - yy1 + 1);
                float inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);

                if (ovr >= this->nms_threshold)
                {
                    isSuppressed[j] = true;
                }
            }
        }
        // return post_nms;
        int idx_t = 0;
        input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
    }

    vector<BoxInfo> NanoDetPlus::object_bboxes(cv::Mat &ori_image) {

        cv::Mat srcImg;
        cv::cvtColor(ori_image, srcImg, cv::COLOR_BGR2RGB);

        int newh = 0, neww = 0, top = 0, left = 0;
        Mat cv_image = srcImg.clone();
        Mat dst = this->resize_image(srcImg, &newh, &neww, &top, &left);
        if(!isQuant){
            this->normalize(dst); //模型为量化模型不需要归一化
        }

        TfliteInterface interface;
        float* model_input = new float[dst.rows*dst.cols*dst.channels()];
        float* model_output = new float[this->output_size];
        interface.infer(model_file, model_input, dst, model_output);

        //generate proposals
        vector<BoxInfo> generate_boxes;
        generate_proposal(generate_boxes, model_output);

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        nms(generate_boxes);

        float ratioh = (float)cv_image.rows / newh;
        float ratiow = (float)cv_image.cols / neww;
//        cout << "boxes: " << generate_boxes.size() << endl;
        for (size_t i = 0; i < generate_boxes.size(); ++i)
        {
            int xmin = (int)max((generate_boxes[i].x1 - left)*ratiow, 0.f);
            int ymin = (int)max((generate_boxes[i].y1 - top)*ratioh, 0.f);
            int xmax = (int)min((generate_boxes[i].x2 - left)*ratiow, (float)cv_image.cols);
            int ymax = (int)min((generate_boxes[i].y2 - top)*ratioh, (float)cv_image.rows);

            generate_boxes[i].x1 = static_cast<float>(xmin);
            generate_boxes[i].y1 = static_cast<float>(ymin);
            generate_boxes[i].x2 = static_cast<float>(xmax);
            generate_boxes[i].y2 = static_cast<float>(ymax);
        }
        return generate_boxes;
        LOG(INFO) << "the detector tflite example finished" << "\n";
    }

    void NanoDetPlus::detect(Mat& srcimg)
    {
        int newh = 0, neww = 0, top = 0, left = 0;
        Mat cv_image = srcimg.clone();
        Mat dst = this->resize_image(cv_image, &newh, &neww, &top, &left);
        if(!isQuant){
            this->normalize(dst); //模型为量化模型不需要归一化
        }

        TfliteInterface interface;
        float* model_input = new float[dst.rows * dst.cols * dst.channels()];
        float* model_output = new float[this->output_size];
        interface.infer(model_file, model_input, dst, model_output);

        //generate proposals
        vector<BoxInfo> generate_boxes;
        generate_proposal(generate_boxes, model_output);

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
        nms(generate_boxes);
        float ratioh = (float)cv_image.rows / newh;
        float ratiow = (float)cv_image.cols / neww;
        cout << "boxes: " << generate_boxes.size() << endl;
        for (size_t i = 0; i < generate_boxes.size(); ++i)
        {
            int xmin = (int)max((generate_boxes[i].x1 - left)*ratiow, 0.f);
            int ymin = (int)max((generate_boxes[i].y1 - top)*ratioh, 0.f);
            int xmax = (int)min((generate_boxes[i].x2 - left)*ratiow, (float)cv_image.cols);
            int ymax = (int)min((generate_boxes[i].y2 - top)*ratioh, (float)cv_image.rows);
            rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
            string label = format("%.2f", generate_boxes[i].score-1.0);
            label = this->class_names[generate_boxes[i].label] + ":" + label;
            putText(srcimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        }
    }

    vector<string> NanoDetPlus::get_class_names() {
        return this->class_names;
    }
};




