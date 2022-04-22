#!/usr/bin/env python3
from __future__ import print_function
from sqlite3 import Time

#import ros message lib
import roslib
from std_msgs.msg import String
from std_msgs.msg import Bool

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from mavros_msgs.msg import State

#ros lib
import rospy
import message_filters

#system
import sys
import os

#opencv lib
import cv2
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs

#tkinter for GUI
import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk

#common lib
import numpy as np
from ast import literal_eval
import time
from playsound import playsound

roslib.load_manifest('human_tracking')

print("OpenCV Version: ", cv2.__version__)

detection_flag = True

#get 3d coordinates
depth_enable = False
def getCoordinate(x, y, camera_depth, cameraInfo):
    if depth_enable:
        _intrinsics = rs.intrinsics()
        _intrinsics.width = cameraInfo.width
        _intrinsics.height = cameraInfo.height
        _intrinsics.ppx = cameraInfo.K[2]
        _intrinsics.ppy = cameraInfo.K[5]
        _intrinsics.fx = cameraInfo.K[0]
        _intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        _intrinsics.model  = rs.distortion.none
        #_intrinsics.coeffs = [i for i in cameraInfo.D]

        result = []
        #result[0]: right, result[1]: down, result[2]: forward
        return (rs.rs2_deproject_pixel_to_point(intrin=_intrinsics, pixel=[x, y], depth=camera_depth[x][y]))
    else:
        return 0.00

class Mask_Detection:
    def __init__(self) -> None:
        #load mask detection model and config file
        model_weights = os.path.expanduser(rospy.get_param("/mask_detection/mask_model_weights"))
        model_cfg = os.path.expanduser(rospy.get_param("/mask_detection/mask_model_cfg"))
        #model_classname = os.path.expanduser(rospy.get_param("/mask_detection/mask_model_classname"))

        #initial darknet for yolo
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

        #use gpu improve performance 
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        #label of result
        self.classes = ["with mask", "without mask"]

        print(model_weights, "loaded.")
        print(model_cfg, "loaded.")

    def detection(self, cv_image, boxes, centers, confidences, class_ids=[], mask_detect=False):
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        filtered_centers = []
        maskDetected = mask_detect

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            numPeople = 0
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
                        if confidence > 0.5:
                            # Get box Coordinates
                            filtered_boxes.append(boxes[i])
                            filtered_confidences.append(confidences[i])
                            filtered_centers.append(centers[i])
                            filtered_class_ids.append(class_id)
                            if class_id == 1:
                                maskDetected = True
                                print("people without mask detected")
                            if class_id == 0:
                                maskDetected = False
                                print("people wear mask")

        return maskDetected, filtered_boxes, filtered_centers, filtered_confidences, filtered_class_ids


class Human_Detection:
    def __init__(self) -> None:
        #load human detection model and config file
        model_weights = os.path.expanduser(rospy.get_param("/mask_detection/human_model_weights"))
        model_cfg = os.path.expanduser(rospy.get_param("/mask_detection/human_model_cfg"))
        #model_classname = os.path.expanduser(rospy.get_param("/mask_detection/human_model_classname"))

        #initial darknet for yolo
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

        #use gpu to improve performance
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        #label of result
        self.classes = ["people"]
        self.class_id = 0

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

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.2:
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
                    class_ids.append(self.class_id)

        return boxes, centers, confidences, class_ids

class Human_Tracking_Node:
    def __init__(self):
        self.bridge = CvBridge()

        #create object of mask and human detection class
        self.Mask_Detection = Mask_Detection()
        self.Human_Detection = Human_Detection()

        #color_topic = rospy.get_param("/mask_detection/color_topic")
        #self.image_sub = message_filters.Subscriber(rospy.get_param("/mask_detection/color_topic"), CompressedImage)#rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback, queue_size=1)
        #self.depth_sub = message_filters.Subscriber(rospy.get_param("/mask_detection/depth_topic"), Image)#rospy.Subscriber("/camera/depth/image_rect_raw/compressed", CompressedImage, self.depth_callback, queue_size=1)
        #self.camera_info = message_filters.Subscriber(rospy.get_param("/mask_detection/depth_camera_info"), CameraInfo)

        #publish detection result to user control GUI
        self.output_pub = rospy.Publisher("/human_tracking/mask_detection/boxes", String, queue_size=10)

        #publish fps
        self.fps_pub = rospy.Publisher("/human_tracking/mask_detection/fps", String, queue_size=10)

        #publish target position
        self.target_pub = rospy.Publisher("/human_tracking/mask_detection/target", Twist, queue_size=10)
        self.target = Twist()

        self.rate = 1

        #sync three topic
        #self.timeSync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info], 10, 10)
        #self.timeSync.registerCallback(self.callback)

        #get image from camera
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback, queue_size=1)

        #store detection result
        self.boxes = []
        self.centers = []
        self.confidences = []
        self.class_ids = []


        self.fps = np.zeros(30)
        self.avg_fps = 0
        self.frame = 0
        self.track_target = False
        self.mask_detection_period = 15

    #When there is a new image updated, call this function
    def callback(self, image):
        if(self.frame > 3600):
            self.frame=0
        self.frame += 1     

        try:
            #convert ros compressed image message to cv2 image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image)
            #cv_depth = self.bridge.imgmsg_to_cv2(depth)

        except CvBridgeError as e:
            print(e)

        start_time = time.time()
        #detection for this frame#####################################
        if (self.frame%self.mask_detection_period == 0):
            #detect mask with a period to reduce the use of computing resources
            self.track_target,self.boxes, self.centers, self.confidences, self.class_ids  = self.Mask_Detection.detection(cv_image, *self.Human_Detection.detection(cv_image), self.track_target)
            
            #if people without mask detected, change label of people detection
            if(self.track_target == True):
                self.Human_Detection.class_id = 1
            else:
                self.Human_Detection.class_id = 0
        else:
            #detect only people
            self.boxes, self.centers, self.confidences, self.class_ids = self.Human_Detection.detection(cv_image)
        ##############################################################
        #get fps
        end_time = time.time()
        self.fps[self.frame%30] = 1/(end_time-start_time)
        for x in range(30):
                self.avg_fps += self.fps[x]
        self.avg_fps = self.avg_fps/30

        #publish detection result and performance
        self.output_pub.publish(str(self.boxes)+'|'+str(self.confidences)+'|'+str(self.class_ids))      
        self.fps_pub.publish(str("{:.2f}".format(self.avg_fps)))

        #filtering detection result
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        target_detected = False
        if len(indices) > 0:
            for i in indices.flatten():
                x, y = self.centers[i]

                #if people without mask detected, tracking that people
                if(self.track_target):
                    self.target.linear.x = x
                    self.target.linear.y = self.boxes[i][3]
                    self.target.linear.z = 0
                    self.target_pub.publish(self.target)
                    target_detected = True

        #if people without mask detected but people out of camera, send 0 to notity offboard mode to find that people
        #if tracked people wear mask detected, send 0 to stop tracking
        if target_detected == False:
            self.target.linear.x = 0
            self.target.linear.y = 0
            self.target.linear.z = 0
            self.target_pub.publish(self.target)

def main(args):
    #create a ros node name "human_tracking"
    rospy.init_node('mask_detection', anonymous=True)

    #create new class and start listening new image data
    tracking = Human_Tracking_Node()

    #set publish frequance
    rospy.Rate = 1/30

    #rospy.spin()
    while not rospy.is_shutdown():
        rospy.sleep(rospy.Rate)

if __name__ == '__main__':
    main(sys.argv)