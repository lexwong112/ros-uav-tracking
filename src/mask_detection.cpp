#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <numeric>

#include "utils/yoloNet.h"

using namespace std;
using namespace cv;
using namespace dnn;

float confThreshold = 0.5;
float nmsThreshold = 0.4;
int inpWidth = 416;
int inpHeight = 416;

// static string cfg_path = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4-custom.cfg";
// static string weight_path = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4.weights";
// static string classid_path = "/home/ho/catkin_ws/src/human_tracking/config/human_classes.txt";

// static yoloNet yolo = yoloNet(cfg_path, weight_path, classid_path, 1280, 800, 0.5);
// static bool detected_flag;
// static bool obstacle = 0;
// static bool identify_flag = true;


void postprocess(cv::Mat& frame,const vector<cv::Mat>& outs){
    vector<int> classIds;//储存识别类的索引
    vector<float> confidences;//储存置信度
    vector<cv::Rect> boxes;//储存边框
    for(size_t i=0;i<outs.size();i++){
    //从网络输出中扫描所有边界框
    //保留高置信度选框
    //目标数据data:x,y,w,h为百分比，x,y为目标中心点坐标
        float* data = (float*)outs[i].data;
        for(int j=0;j<outs[i].rows;j++,data+=outs[i].cols){
            cv::Mat scores = outs[i].row(j).colRange(5,outs[i].cols);
            cv::Point classIdPoint;
            double confidence;//置信度
            //取得最大分数值与索引
            cv::minMaxLoc(scores,0,&confidence,0,&classIdPoint);
            if(confidence>confThreshold){
                int centerX = (int)(data[0]*frame.cols);
                int centerY = (int)(data[1]*frame.rows);
                int width = (int)(data[2]*frame.cols);
                int height = (int)(data[3]*frame.rows);
                int left = centerX-width/2;
                int top = centerY-height/2;
                classIds.push_back(classIdPoint.x);
                       confidences.push_back((float)confidence);
                       boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    //低置信度
    vector<int> indices;//保存没有重叠边框的索引
    //该函数用于抑制重叠边框
    cv::dnn::NMSBoxes(boxes,confidences,confThreshold,nmsThreshold,indices);
    for(size_t i=0;i<indices.size();i++){
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        //drawPred(classIds[idx],confidences[idx],box.x,box.y,
        //box.x+box.width,box.y+box.height,frame);
    }
}


class Mask_Detection
{
    private:

        String cfg_path = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4-custom.cfg";
        String weight_path = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4.weights";
        //String classid_path = "/home/ho/catkin_ws/src/human_tracking/config/human_classes.txt";
        Net net;

        int width = 1280;
        int height = 800;
        vector<String> classes;
        vector<String> outputNames;

        vector<float> confidences; // from 0 to 1
        vector<int> classIds;  //0, 1, 2, 3
        vector<Rect> boundingBoxes;  // Rect class



    public:
        Mask_Detection()
        {
            classes.push_back("with_mask");
            classes.push_back("without_mask");

            this->net = readNet(cfg_path, weight_path);
            this->net.setPreferableBackend(DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(DNN_TARGET_CUDA);
        
            outputNames = net.getUnconnectedOutLayersNames();

            
        }

        void drawBoudingBox(Mat &img)
        {
            for (int i = 0; i < this->objects.size(); i++)
            {
                this->inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
                const auto color = colors[i % NUM_COLORS];

                rectangle(img, objects[i].draw_box, color, 3);
                rectangle(img, objects[i].depth_box, color, 2);
                rectangle(img, objects[i].env_box.out, color, 1);
                rectangle(img, objects[i].env_box.top, Scalar(198,227,171), -1);
                rectangle(img, objects[i].env_box.bottom, Scalar(198,227,171), -1);
                rectangle(img, objects[i].env_box.left, Scalar(198,227,171), -1);
                rectangle(img, objects[i].env_box.right, Scalar(198,227,171), -1);
                //Create the label Text
                String labelText = format("%.2f", objects[i].confidence);
                labelText = objects[i].classId + ":" + labelText;
                //Draw the label text on the image
                int baseline;
                Size labelSize = getTextSize(labelText, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                objects[i].draw_box.y = max(objects[i].draw_box.y , labelSize.height);
                rectangle(img, Point(objects[i].draw_box.x, objects[i].draw_box.y  - labelSize.height) , Point (objects[i].draw_box.x + labelSize.width, objects[i].draw_box.y  + baseline), color, FILLED);
                putText(img, labelText, objects[i].draw_box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0));
            }

            this->total_end = std::chrono::steady_clock::now();
            float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
            float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
            std::ostringstream stats_ss;
            stats_ss << std::fixed << std::setprecision(2);
            stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
            auto stats = stats_ss.str();
            int baseline;
            auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(img, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(img, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
            imshow("Output of prediction", img);
            waitKey(60);
            
        }


        void detection(Mat &frame)
        {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame,blob,1/255.0,cv::Size(width,height));
            net.setInput(blob);
            vector<cv::Mat> outs;
            net.forward(outs, outputNames);

            confidences.clear();
            classIds.clear();
            boundingBoxes.clear();

            for (auto & output : outs)
            {
                for(int j = 0; j<output.rows; j++)  
                {
                    auto objectnessPrediction = output.at <float> (j, 4);

                    if (objectnessPrediction >= this->confThreshold) // 目标置信度大于confThreshold
                    {
                        Mat classPrediction = output.row(j).colRange(5, output.cols); // class prob part
                        Point maxPoint; // class number corresponding to maximum class probaility
                        double maxVal; //maximum class probability
                        minMaxLoc(classPrediction, 0, &maxVal, 0, &maxPoint); //find the maximum class probability
                        auto x = output.at <float> (j, 0) * frame.cols;
                        auto y = output.at <float> (j, 1) * frame.rows;
                        auto w = output.at <float> (j, 2) * frame.cols;
                        auto h = output.at <float> (j, 3) * frame.rows;

                        confidences.push_back(maxVal); // probability maybe not larger than zero
                        classIds.push_back(maxPoint.x); // class number in obj.names
                        boundingBoxes.push_back(Rect(x, y, w, h));
                    }
                }
            }

            cout<<boxes;
            std::vector<int> indices;
            NMSBoxes(boundingBoxes, confidences, this->confThreshold, this->nmsThreshold, indices);

            postprocess(frame, outs);
            cv::putText(frame,"text",cv::Point(0,15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,255));
        
            Mat detecteFrame;
            frame.convertTo(detecteFrame,CV_8U);
            cv::imshow("Output",frame);
            cout<<"show img";
        
        }
};



class Human_Tracking
{
    private:
        
        ros::NodeHandle nh;

        ros::Subscriber image_sub = nh.subscribe("/camera/color/image_raw/compressed", 10, &Human_Tracking::callback, this);

        Mask_Detection mask_detection;
    public:
        void callback(const sensor_msgs::CompressedImageConstPtr & rgb)
        {
            cv::Mat image_rgb;

            try
            {
                image_rgb = cv::imdecode(cv::Mat(rgb->data),1);
            }
            catch(cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            mask_detection.detection(image_rgb);

        }
};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "human_tracking");
    
    Human_Tracking tracking;

    ros::Rate loop_rate(20);

    while(ros::ok())
    {
        ros::spinOnce();    
        loop_rate.sleep();
    }

  return 0;
    
}