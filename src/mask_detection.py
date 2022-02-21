#!/usr/bin/env python3
from __future__ import print_function
from sqlite3 import Time

import numpy as np
from logging import Logger

import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo

import rospy
import sys

import glob

import sys

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
import message_filters

roslib.load_manifest('human_tracking')

print(cv2.__version__)

detection_flag = True

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]


def remove_suffix(text, prefix):
    if text.endswith(prefix):
        return text[:len(text) - len(prefix)]


def project_2d_to_3d(depth, center):
    #depth_profile = depth.get_profile()
    cvs_profile = rs.video_stream_profile(depth)
    color_intrinsic = cvs_profile.get_intrinsics()
    #camera_coordiantes = []

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

import struct

def convert_depth_to_phys_coord_using_realsense(centers, camera_depth, cameraInfo):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]

    result = []
    for x, y in centers:
        if x<480 and y<848:
            result.append(rs.rs2_deproject_pixel_to_point(intrin=_intrinsics, pixel=[x, y], depth=camera_depth[x][y]))

    #result[0]: right, result[1]: down, result[2]: forward
    return result#[2], -result[0], -result[1]

class Mask_Detection:
    def __init__(self) -> None:
        model_weights = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-obj_final.weights"
        model_cfg = "/home/ho/catkin_ws/src/human_tracking/config/mask_yolov4-custom.cfg"
        model_classname = "/home/ho/catkin_ws/src/human_tracking/config/mask_classes.txt"

        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["with mask", "without mask"]
        #with open(model_classname, "r") as file_object:
        #    for class_name in file_object.readlines():
        #        class_name = class_name.strip()
        #        self.classes.append(class_name)

        print(model_weights, "loaded.")
        print(model_cfg, "loaded.")

    def detection_2(self, cv_image):
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
                if confidence > 0.01:
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
                    #print(rospy.get_rostime(), ": people without mask detected")

        #show_detected_image(self, cv_image, boxes, confidences, class_ids)
        return boxes, centers, confidences, class_ids

    def detection(self, cv_image, boxes, centers, confidences, class_ids=[]):
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_centers = []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            numPeople = 0
            maskDetected = False
            for i in indices.flatten():
                numPeople += 1
                x, y, w, h = boxes[i]

                try:
                    cropped_image = cv_image[y:y + h, x:x + w]
                    img = cv2.resize(cropped_image, (cv_image.shape[:2]), interpolation=cv2.INTER_AREA)
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
                        if confidence > 0.6:
                            # Get box Coordinates
                            filtered_boxes.append(boxes[i])
                            filtered_confidences.append(confidences[i])
                            filtered_centers.append(centers[i])
                            filtered_class_ids.append(class_id)
                            if class_id == 1:
                                maskDetected = True
                            
            print(numPeople, " people detected.")
            if maskDetected:
                print("people without mask detected")

        #show_detected_image(self, cv_image, filtered_boxes, filtered_confidences, filtered_class_ids)
        return filtered_boxes, filtered_centers, filtered_confidences, filtered_class_ids
        
def show_detected_image(self, image, boxes, confidences, class_ids, windowName = "Output"):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                #get coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #darw bounding box
                color = (255, 1, 12)#[int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)             
                #put text
                text = "{}: {:.2f}".format(self.classes[class_ids[i]], confidences[i])
                
                cv2.rectangle(image, (x, y-28), (x + w, y), (255,255,255), -1)
                cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, lineType=cv2.LINE_AA)

        cv2.imshow(windowName, image)
        cv2.waitKey(3)
   
class Human_Detection:
    def __init__(self) -> None:
        model_weights = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4.weights"
        model_cfg = "/home/ho/catkin_ws/src/human_tracking/config/human_yolov4-custom.cfg"
        model_classname = "/home/ho/catkin_ws/src/human_tracking/config/human_classes.txt"
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.classes = ["people"]
        #with open(model_classname, "r") as file_object:
        #    for class_name in file_object.readlines():
        #        class_name = class_name.strip()
        #        self.classes.append(class_name)

        print(model_weights, "loaded.")
        print(model_cfg, "loaded.")

    def detection(self, cv_image):
        confidences = []
        boxes = []
        centers = []
        class_ids = []

        height, width = cv_image.shape[:2]

        blob = cv2.dnn.blobFromImage(cv_image, 1.0 / 255.0, (416, 416), None, True, False)
        output_layers = self.net.getUnconnectedOutLayersNames()

        self.net.setInput(blob)
        outs = self.net.forward(output_layers)
        numPeople = 0
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.6:
                    c_x = int(detection[0] * width)
                    c_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(c_x - w / 2)
                    y = int(c_y - h / 2)

                    box = detection[0:4] * np.array([width, height, width, height])

                    boxes.append([x, y, w, h])
                    centers.append([c_x, c_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    numPeople+=1

        #print(numPeople, " people detected")

        
        return boxes, centers, confidences, class_ids

class Human_Tracking_Node:
    def __init__(self):
        self.bridge = CvBridge()
        self.Mask_Detection = Mask_Detection()
        self.Human_Detection = Human_Detection()
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)#rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback, queue_size=1)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_rect_raw', Image)#rospy.Subscriber("/camera/depth/image_rect_raw/compressed", CompressedImage, self.depth_callback, queue_size=1)
        self.output_pub = rospy.Publisher("/human_tracking/mask_detection/boxes", String, queue_size=10)
        self.camera_info = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo)
        #self.imageDisplayer = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.only_image_callback, queue_size=1)

        self.rate = 1
        self.timeSync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info], 10, 10)
        self.timeSync.registerCallback(self.callback)
        self.boxes = []
        self.centers = []
        self.confidences = []
        self.class_ids = []
        self.frame = 0

    def callback(self, image, depth, camera_info):
        #self.image_sub.unregister()
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #np_arr = np.frombuffer(image.data, np.uint8)
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image)#np_arr, cv2.IMREAD_COLOR) 
            #np_arr = np.frombuffer(depth.data, np.uint8)
            cv_depth = self.bridge.imgmsg_to_cv2(depth, depth.encoding)
        except CvBridgeError as e:
            print(e)

        #boxes, centers, confidences, class_ids = self.Human_Detection.detection(cv_image) #well detect for human but show many boxes on same person
        #show_detected_image(self.Mask_Detection, cv_image, boxes, confidences, class_ids, "human detection")
        self.boxes, self.centers, self.confidences, self.class_ids = self.Mask_Detection.detection(cv_image, *self.Human_Detection.detection(cv_image)) #very lag and delay
        #boxes, centers, confidences, class_ids = self.Mask_Detection.detection_2(cv_image)
        #show_detected_image(self.Mask_Detection, cv_image, boxes, confidences, class_ids, "mask detection")
        
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback, queue_size=1)
        # 2D to 3D
        result = convert_depth_to_phys_coord_using_realsense(self.centers, cv_depth, camera_info)
        print(result)


    def boxesPublisher(self):
        self.output_pub.publish(str(self.boxes)+'|'+str(self.confidences)+'|'+str(self.class_ids))

    def only_image_callback(self, image):
        print("new image")
        #self.image_sub.unregister()
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            np_arr = np.frombuffer(image.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        except CvBridgeError as e:
            print(e)

        #boxes, centers, confidences, class_ids = self.Human_Detection.detection(cv_image) #well detect for human but show many boxes on same person
        #show_detected_image(self.Mask_Detection, cv_image, boxes, confidences, class_ids, "human detection")
        self.boxes, self.centers, self.confidences, self.class_ids = self.Mask_Detection.detection(cv_image, *self.Human_Detection.detection(cv_image)) #very lag and delay
        #boxes, centers, confidences, class_ids = self.Mask_Detection.detection_2(cv_image)
        #show_detected_image(self.Mask_Detection, cv_image, boxes, confidences, class_ids, "mask detection")
        
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback, queue_size=1)
        # 2D to 3D
        #result = convert_depth_to_phys_coord_using_realsense(centers, depth, camera_info)
        #print(result)

        


def main(args):
    rospy.init_node('human_tracking', anonymous=True)
    tracking = Human_Tracking_Node()

    rospy.Rate = 1/30
    #rospy.spin()
    while not rospy.is_shutdown():
        tracking.boxesPublisher()
        #rospy.spinOnce()
        rospy.sleep(rospy.Rate)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)