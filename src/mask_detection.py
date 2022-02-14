#!/usr/bin/env python3
from __future__ import print_function
from sqlite3 import Time

import numpy as np
from logging import Logger

import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image

import rospy
import sys

import glob

import sys


sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.remove('/home/ho/catkin_ws/devel/lib/python3/dist-packages')
print("SYS.PATH: ", sys.path)

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs

roslib.load_manifest('human_tracking')

print(cv2.__version__)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]


def remove_suffix(text, prefix):
    if text.endswith(prefix):
        return text[:len(text) - len(prefix)]


def project_2d_to_3d(color, depth, center):
    color_profile = color.get_profile()
    cvs_profile = rs.video_stream_profile(color_profile)
    color_intrinsic = cvs_profile.get_intrinsics()
    camera_coordiantes = []

    if len(center) != 0:
        dis = depth.get_distance(center[0], center[1])
        print("dis", dis)
        result = rs.rs2_deproject_pixel_to_point(color_intrinsic,
                                                 center,
                                                 dis)
        # result[0]: right
        # result[1]: down
        # result[2]: forward
        print(result[2], -result[0], -result[1])
        camera_coordiantes.append([result[2], -result[0], -result[1]])
    #
    return camera_coordiantes

class Mask_Detection:
    def __init__(self) -> None:
        model_weights = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-obj_final.weights"
        model_cfg = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-custom.cfg"
        model_classname = "/home/ho/catkin_ws/src/human_tracking/config/mask_classes.txt"

        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = []
        with open(model_classname, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

    def detection(self, cv_image, boxes, centers, confidences):
        detection_count = len(boxes)

        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_centers = []

        for i in range(detection_count):
            x, y, w, h = boxes[i]

            try:
                cropped_image = cv_image[y:y + h, x:x + w]
                img = cv2.resize(cropped_image, (1500, 750), interpolation=cv2.INTER_AREA)
            except:
                continue

            blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True, False)
            output_layers = self.net.getUnconnectedOutLayersNames()
            self.net.setInput(blob)
            outs = self.net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 1 and confidence > 0.6:
                        # Get box Coordinates
                        filtered_boxes.append(boxes[i])
                        filtered_confidences.append(confidences[i])
                        filtered_centers.append(centers[i])
                        filtered_class_ids.append(class_id)
                        
                        print(rospy.get_rostime(), ": people without mask detected")

        image = cv_image.copy()
        indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_confidences, 0.5, 0.4)
        for i in indices:
            box = filtered_boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # camera_coordinates = project_2d_to_3d(color, filtered_centers[i])
            cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
            cv2.putText(image, self.classes[filtered_class_ids[i]], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(image, str(filtered_confidences[i]), (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                        2)

        #cv2.imwrite(r"D:\POLYU\YEAR_4\CAPSTONE\SEM_2\Output\image2.jpg", image)
        #self.colorwriter.write(image)
        cv2.imshow("Image window", image)
        cv2.waitKey(3)

        return filtered_boxes, filtered_centers, filtered_confidences,
        
        

class Human_Detection:
    def __init__(self) -> None:
        model_weights = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4.weights"
        model_cfg = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4-custom.cfg"
        model_classname = "/home/ho/catkin_ws/src/human_tracking/config/human_classes.txt"
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = []
        with open(model_classname, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        

    def detection(self, cv_image):
        # print(self.image)
        confidences = []
        boxes = []
        centers = []
        class_ids = []

        height, width = cv_image.shape[:2]

        # layer_names = self.net.getLayerNames()
        blob = cv2.dnn.blobFromImage(cv_image, 1.0 / 255.0, (416, 416), None, True, False)
        output_layers = self.net.getUnconnectedOutLayersNames()
        # print(self.image.shape)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.6:
                    # Object detected
                    # print(class_id)
                    c_x = int(detection[0] * width)
                    c_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(c_x - w / 2)
                    y = int(c_y - h / 2)

                    boxes.append([x, y, w, h])
                    centers.append([c_x, c_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # image = self.image.copy()
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # for i in indices:
        #     box = boxes[i]
        #     left = box[0]
        #     top = box[1]
        #     width = box[2]
        #     height = box[3]
        #     # camera_coordinates = project_2d_to_3d(color, centers[i])
        #     project_2d_to_3d(color, centers[i])
        #     cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
        #     cv2.putText(image, self.classes[class_ids[i]], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     cv2.putText(image, confidences[i], (left, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # cv2.imwrite(r"D:\POLYU\YEAR_4\CAPSTONE\SEM_2\Output\image.jpg", image)
        # self.colorwriter.write(image)

        return boxes, centers, confidences

class Detection:
    def __init__(self, cv2_image, model_weights, model_cfg, model_classes):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        # self.model = self.load_model(model_name)\
        self.image = cv2_image
        self.height, self.width = self.image.shape[:2]
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_GPU)

        self.classes = []
        with open(model_classes, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

    def human_detection(self):
        # print(self.image)
        confidences = []
        boxes = []
        centers = []
        class_ids = []
        # layer_names = self.net.getLayerNames()
        blob = cv2.dnn.blobFromImage(self.image, 1.0 / 255.0, (416, 416), None, True, False)
        output_layers = self.net.getUnconnectedOutLayersNames()
        # print(self.image.shape)
        self.net.setInput(blob)
        outs = self.net.forward(output_layers)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.6:
                    # Object detected
                    # print(class_id)
                    c_x = int(detection[0] * self.width)
                    c_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)

                    # Rectangle coordinates
                    x = int(c_x - w / 2)
                    y = int(c_y - h / 2)

                    boxes.append([x, y, w, h])
                    centers.append([c_x, c_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # image = self.image.copy()
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # for i in indices:
        #     box = boxes[i]
        #     left = box[0]
        #     top = box[1]
        #     width = box[2]
        #     height = box[3]
        #     # camera_coordinates = project_2d_to_3d(color, centers[i])
        #     project_2d_to_3d(color, centers[i])
        #     cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
        #     cv2.putText(image, self.classes[class_ids[i]], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     cv2.putText(image, confidences[i], (left, top+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # cv2.imwrite(r"D:\POLYU\YEAR_4\CAPSTONE\SEM_2\Output\image.jpg", image)
        # self.colorwriter.write(image)

        return boxes, centers, confidences

    def mask_detection(self, boxes, centers, confidences):
        detection_count = len(boxes)

        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_centers = []

        for i in range(detection_count):
            x, y, w, h = boxes[i]

            try:
                cropped_image = self.image[y:y + h, x:x + w]
                img = cv2.resize(cropped_image, (1500, 750), interpolation=cv2.INTER_AREA)
            except:
                continue

            blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (416, 416), None, True, False)
            output_layers = self.net.getUnconnectedOutLayersNames()
            self.net.setInput(blob)
            outs = self.net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 1 and confidence > 0.6:
                        # Get box Coordinates
                        filtered_boxes.append(boxes[i])
                        filtered_confidences.append(confidences[i])
                        filtered_centers.append(centers[i])
                        filtered_class_ids.append(class_id)
                        
                        print(rospy.get_rostime(), ": people without mask detected")

        image = self.image.copy()
        indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_confidences, 0.5, 0.4)
        for i in indices:
            box = filtered_boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # camera_coordinates = project_2d_to_3d(color, filtered_centers[i])
            cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2, 8, 0)
            cv2.putText(image, self.classes[filtered_class_ids[i]], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(image, str(filtered_confidences[i]), (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
                        2)

        #cv2.imwrite(r"D:\POLYU\YEAR_4\CAPSTONE\SEM_2\Output\image2.jpg", image)
        #self.colorwriter.write(image)
        #cv2.imshow("Image window", image)
        #cv2.waitKey(3)

        return filtered_boxes, filtered_centers, filtered_confidences,

class image_converter:

    def __init__(self):
        # self.image_pub = rospy.Publisher("result topic",data type, queue_size=10)
        self.logger_pub = rospy.Publisher("/log/mask_detection", String, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", String, self.depth_callback)
        self.output_pub = rospy.Publisher("/mask_detection/color/result", String, queue_size=10)
        self.rate = 1

        self.Mask_Detection = Mask_Detection()
        self.Human_Detection = Human_Detection()

    def color_callback(self, data):  # when there is a new image data
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #boxes, centers, confidences = self.Human_Detection.detection(cv_image)
        boxes, centers, confidences = self.Mask_Detection.detection(cv_image, *self.Human_Detection.detection(cv_image))

        #cv2.imshow("Image window", cv_image)  # show original image
        #cv2.waitKey(3)

        ################################
        #######  mask detection  #######
        #
        #

        # Convert to array
        #cv_image = np.asanyarray(cv_image.get_data())
        # Convert from BGR to RGB
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Input filenames path for Human Detection
        #model_human_weights = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4.weights"
        #model_human_cfg = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4-custom.cfg"
        #model_human_classname = "/home/ho/catkin_ws/src/human_tracking/config/human_classes.txt"

        # Detect Human
        #detection = Detection(cv_image, model_human_weights, model_human_cfg, model_human_classname)
        #boxes, centers, confidences = detection.human_detection()

        # Input filenames path for Human-Mask Detection
        #model_mask_weights = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-obj_final.weights"
        #model_mask_cfg = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-custom.cfg"
        #model_mask_classname = "/home/ho/catkin_ws/src/human_tracking/config/mask_classes.txt"

        # Detect human NOT wearing mask
        #detection = Detection(cv_image, model_mask_weights, model_mask_cfg, model_mask_classname)
        #boxes, centers, confidences = detection.mask_detection(boxes, centers, confidences)



        # 2D to 3D
        #result = project_2d_to_3d(cv_image, cv_depth, centers)

        #
        # self.image_pub.publish(result)#
        ################################

    def depth_callback(self, data):  # when there is a new image data
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #cv2.imshow("Image window", cv_image)  # show original image
        #cv2.waitKey(3)

        ################################
        #######  mask detection  #######
        #
        #
        #
        # self.image_pub.publish(result)#
        ################################

        self.output_pub.publish("result")

    def logger(self):
        self.logger_pub.publish("test")


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.Rate = 1
    while not rospy.is_shutdown():
        rospy.sleep(rospy.Rate)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)