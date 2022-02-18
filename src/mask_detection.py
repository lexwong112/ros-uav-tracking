#!/usr/bin/env python3
from __future__ import print_function
from sqlite3 import Time

import numpy as np
from logging import Logger

import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import rospy
import sys

import glob

import sys

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

        print(model_weights, "loaded.")
        print(model_cfg, "loaded.")

    def detection(self, cv_image, boxes, centers, confidences):
        #detection_count = len(boxes)
        
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_centers = []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #for i in range(detection_count):
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            try:
                cropped_image = cv_image[y:y + h, x:x + w]
                img = cv2.resize(cropped_image, (1280, 720), interpolation=cv2.INTER_AREA)
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
                        np.append(filtered_boxes, (boxes[i]))
                        filtered_confidences.append(confidences[i])
                        filtered_centers.append(centers[i])
                        filtered_class_ids.append(class_id)
                        
                        print(rospy.get_rostime(), ": people without mask detected")

        print(filtered_centers)  
        show_detected_image(self, cv_image, filtered_boxes, filtered_confidences, filtered_class_ids)
        return filtered_boxes, filtered_centers, filtered_confidences   
        
def show_detected_image(self, image, boxes, confidences, class_ids):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                #get coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #darw bounding box and label
                color = 112#[int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
                text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
                cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, lineType=cv2.LINE_AA)
                
        cv2.imshow("Image window", image)
        cv2.waitKey(3)
        
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

        print(model_weights, "loaded.")
        print(model_cfg, "loaded.")

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
        numPeople = 0
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

                    box = detection[0:4] * np.array([width, height, width, height])

                    boxes.append([x, y, w, h])
                    centers.append([c_x, c_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    numPeople+=1

        print(numPeople, " people detected")

        show_detected_image(self, cv_image, boxes, confidences, class_ids)
        return boxes, centers, confidences

class image_converter:

    def __init__(self):
        # self.image_pub = rospy.Publisher("result topic",data type, queue_size=10)
        self.bridge = CvBridge()
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)#, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback)#, queue_size=1)
        self.output_pub = rospy.Publisher("/human_tracking/mask_detection/result/centers", String, queue_size=10)
        self.rate = 1

        self.Mask_Detection = Mask_Detection()
        self.Human_Detection = Human_Detection()

    def color_callback(self, data):
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            np_arr = np.frombuffer(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        except CvBridgeError as e:
            print(e)

        #boxes, centers, confidences = self.Human_Detection.detection(cv_image) #well detect for human but show many boxes on same person
        boxes, centers, confidences = self.Mask_Detection.detection(cv_image, *self.Human_Detection.detection(cv_image)) #very lag and delay

        print(centers)

        # 2D to 3D
        #result = project_2d_to_3d(cv_image, cv_depth, centers)

        #
        # self.image_pub.publish(result)#
        ################################

    def depth_callback(self, data):
        pass

def main(args):
    ic = image_converter()
    rospy.init_node('human_tracking', anonymous=True)
    rospy.Rate = 1
    while not rospy.is_shutdown():
        rospy.sleep(rospy.Rate)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)