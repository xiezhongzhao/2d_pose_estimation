/**
 * @Author:  xiezhongzhao
 * @Email:   2234309583@qq.com
 * @Data:    2022/9/20 9:56
 * @Version: 1.0
**/

#include <pose_estimation/nanodet.h>
#include <pose_estimation/pose2d.h>
#include <pose_estimation/euro_filter.h>

void detectImg(Info& info){

    string nanodet_model_file = info.nanodet_path;
    string mobilev2_model_file = info.pose_path;
    string coco_name = info.class_path;
    string image_in_path = info.img_in_path;
    string image_out_path = info.img_out_path;

    cv::Mat ori_img = cv::imread(image_in_path);

    // nanodet人体检测调用
    detector::NanoDetPlus detect(nanodet_model_file, coco_name, 288, 512, 0.3, 0.4);
    // mobilenetv2人体姿态估计
    pose::Pose2d pose2D(mobilev2_model_file, 224, 224);
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
        pose2D.drawSkeleton(ori_img, rec_joints, xmin, ymin);

        // 画出人体目标框和对应的人体姿态关键点
        rectangle(ori_img, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 1);
        string label = format("%.2f", boxes[i].score);
        vector<string> class_names = detect.get_class_names();
        label = class_names[boxes[i].label] + ":" + label;
        putText(ori_img, label, Point(xmin, ymin - 5),
                FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 0, 255), 2);
    }
    cv::imwrite(image_out_path, ori_img);
    cout << "Finished !!!" << endl;
}

void detectVideo(Info& info){

    string nanodet_model_file = info.nanodet_path;
    string mobilev2_model_file = info.pose_path;
    string coco_name = info.class_path;
    string video_in_path = info.video_in_path;
    string video_out_path = info.video_out_path;

    // nanodet人体检测调用
    detector::NanoDetPlus detect(nanodet_model_file, coco_name, 288, 512, 0.3, 0.4);
    // mobilenetv2人体姿态估计
    pose::Pose2d pose2D(mobilev2_model_file, 224, 224);

    cv::VideoCapture cap(video_in_path);
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return ;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter video(video_out_path,
                          cv::VideoWriter::fourcc('X', '2', '6', '4'),
                          25, Size(frame_width, frame_height));
    while(true){
        Mat frame;
        cap >> frame;
        if(frame.empty())
            break;

        // 所有人体区域
        vector<detector::BoxInfo> boxes = detect.object_bboxes(frame);
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            int xmin = (int)boxes[i].x1;
            int ymin = (int)boxes[i].y1;
            int xmax = (int)boxes[i].x2;
            int ymax = (int)boxes[i].y2;

            // 裁剪原始图像获得人体区域
            cv::Rect rect(xmin, ymin, xmax-xmin, ymax-ymin);
            cv::Mat img_roi = frame(rect);

            // 人体区域输入姿态模型
            vector<cv::Point> rec_joints = pose2D.getJoints(img_roi);

            // int 转化为 float
            vector<cv::Point2f> rec_joints_f;
            for(const auto& it : rec_joints){
                cv::Point2f tmp;
                tmp.x = static_cast<float>(it.x);
                tmp.y = static_cast<float>(it.y);
                rec_joints_f.emplace_back(tmp);
            }
            // 关键点消抖
            vector<cv::Point2f> rec_joints_filter = filter::Filter2D(rec_joints_f,
                                                                     0.18, 0.70, 30);

            // float 转化为 int
            vector<cv::Point> rec_joints_filter_int;
            for(const auto& it : rec_joints_filter){
                cv::Point tmp;
                tmp.x = static_cast<int>(it.x);
                tmp.y = static_cast<int>(it.y);
                rec_joints_filter_int.emplace_back(tmp);
            }

            // 获得人体姿态关键点，并且投影会原始图像
            pose2D.drawSkeleton(frame, rec_joints_filter_int, xmin, ymin);

            // 画出人体目标框和对应的人体姿态关键点
            rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
            string label = format("%.2f", boxes[i].score);
            vector<string> class_names = detect.get_class_names();
            label = class_names[boxes[i].label] + ":" + label;
            putText(frame, label, Point(xmin, ymin - 5),
                    FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0, 0, 255), 2);
        }

        video.write(frame);
    }
    cap.release();
    cv::destroyAllWindows();
    cout << "Finished !!!" << endl;
}


int main(){

    Info info;
    info.root_path = "/mnt/e/WorkSpace/CPlusPlus/2d_pose_estimation/";
    info.nanodet_path = Info::addRootPath(info.root_path, "model/nanodetplus_person.tflite");
    info.class_path = Info::addRootPath(info.root_path, "model/coco.names");
    info.pose_path = Info::addRootPath(info.root_path,"model/mobilenetv2_224_224_heatmap.tflite");
    info.img_in_path = Info::addRootPath(info.root_path,"data/family.jpg");
    info.img_out_path = Info::addRootPath(info.root_path,"data/family_out.jpg");
    info.video_in_path = Info::addRootPath(info.root_path,"data/black_man.mp4");
    info.video_out_path = Info::addRootPath(info.root_path,"data/black_man_out.avi");

    detectImg(info);
//    detectVideo(info);
    return 0;
}







