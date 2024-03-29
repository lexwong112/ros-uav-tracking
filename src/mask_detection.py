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

#common lib
import numpy as np
from ast import literal_eval
import time
from playsound import playsound

roslib.load_manifest('human_tracking')

print("OpenCV Version: ", cv2.__version__)

detection_flag = True
pi=3.14159
image_width = 640
image_height = 360

#get 3d coordinates use realsense function
depth_enable = True
def getCoordinates(x, y, camera_depth, cameraInfo):
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

#install with 45 degree, custom function, get coordinates from pixels position of image
def getCoordinate(x, y, distance, id=0):
    if(y>image_width/2):
        angle_y = ((y-(image_width/2))/(image_width/2))*45
    elif(y<image_width/2):
        angle_y = -(((image_width/2)-y)/(image_width/2))*45
    elif(y==image_width/2):
        angle_y=0

    if(x>image_height/2):
        angle_x = 45-((x-image_height/2)/(image_height/2))*32.5
    elif(x<image_height/2):
        angle_x = 45+(x/(image_height/2))*32.5
    elif(x==image_height/2):
        angle_x=0

    #minus to change axis direction
    target_y=-distance*np.sin(angle_y*(pi/180))
    projected_distance = distance*np.sin(angle_x*(pi/180))
    target_x=np.sqrt((projected_distance*projected_distance) - (target_y*target_y))

    print("target position ID: ", id,"\nx: ",target_x,"\ny: ",target_y,"\ndistance: ",distance,"\nAngle y: ", angle_y, "\nAngle x: ",angle_x, "\n")
    #for test
    print("center x: ", x,"\ncenter y: ",y)
    return target_x, target_y


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
                                #print("people without mask detected")
                            if class_id == 0:
                                maskDetected = False
                                #print("people wear mask")
        iboxes=[]
        icenters=[]
        iconfidences=[]
        ids=[]
        indices = cv2.dnn.NMSBoxes(filtered_boxes, filtered_confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                iboxes.append(filtered_boxes[i])
                icenters.append(filtered_centers[i])
                iconfidences.append(filtered_confidences[i])
                ids.append(filtered_class_ids[i])             
        return maskDetected, iboxes, icenters, iconfidences, ids


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

        iboxes=[]
        icenters=[]
        iconfidences=[]
        ids=[]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                iboxes.append(boxes[i])
                icenters.append(centers[i])
                iconfidences.append(confidences[i])
                ids.append(class_ids[i])             
        return iboxes, icenters, iconfidences, ids

class Human_Tracking_Node:
    def __init__(self):
        self.bridge = CvBridge()

        #create object of mask and human detection class
        self.Mask_Detection = Mask_Detection()
        self.Human_Detection = Human_Detection()

        #color_topic = rospy.get_param("/mask_detection/color_topic")
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw/compressed", CompressedImage)#rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.color_callback, queue_size=1)
        self.depth_sub = message_filters.Subscriber("/camera/depth_aligned_to_color_and_infra1/image_raw", Image)#rospy.Subscriber("/camera/depth/image_rect_raw/compressed", CompressedImage, self.depth_callback, queue_size=1)
        self.camera_info = message_filters.Subscriber("/camera/depth_aligned_to_color_and_infra1/camera_info", CameraInfo)

        #publish detection result to user control GUI
        self.output_pub = rospy.Publisher("/human_tracking/mask_detection/boxes", String, queue_size=10)

        #publish fps
        self.fps_pub = rospy.Publisher("/human_tracking/mask_detection/fps", String, queue_size=10)

        #publish target position
        self.target_pub = rospy.Publisher("/human_tracking/mask_detection/target", Twist, queue_size=10)
        self.target = Twist()

        #publish target coordinates base on uav.
        self.target_coordinates_pub = rospy.Publisher("/human_tracking/mask_detection/target/coordinates", Twist, queue_size=10)
        self.target_coordinates = Twist()

        #publish message to user control GUI
        self.message_pub = rospy.Publisher("/user_control/message", String, queue_size=10)

        self.rate = 1

        #sync three topic
        self.timeSync = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info], 10, 10)
        self.timeSync.registerCallback(self.callback)

        #get image from camera
        #rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.callback, queue_size=1)

        #store detection result
        self.boxes = []
        self.centers = []
        self.confidences = []
        self.class_ids = []

        #object tracking
        self.multiTracker = cv2.legacy.MultiTracker_create()
        self.center_points_cur_frame = []
        self.tracker_id = 0
        self.tracking_objects = {}
        self.tracker = tracker = cv2.legacy.TrackerKCF_create()

        self.fps = np.zeros(30)
        self.avg_fps = 0
        self.frame = 0
        self.track_target = False
        self.mask_detection_period = 15

    #get angle of target from image
    def getAngle(self, x, y):
        """
        if(y>image_width/2):
            angle_y = (y/image_width)*45
        elif(y<image_width/2):
            angle_y = -(y/image_width)*45
        elif(y==image_width/2):
            angle_y=0

        if(x>image_height/2):
            angle_x = 45-(x/image_height)*32.5
        elif(x<image_height/2):
            angle_x = 45+(x/image_height)*45
        elif(x==image_height/2):
            angle_x=0
        """
        #for D455
        hfov = 90
        vfov = 45
        camera_angle = 45
        angle_x = camera_angle+(vfov/2)-((vfov*x)/image_height)
        angle_y = (hfov*y)/image_height-(hfov/2)#for D455
        return angle_x, angle_y

    #target tracking and obtain unique ID
    def object_tracker(self, frame):
        # Point current frame
        self.center_points_cur_frame = []
        if self.frame%10 == 1:
            boxes = []
            (class_ids, scores, boxes) = od.detect(frame)
            bboxes = copy.copy(self.boxes)
            for box in bboxes:
                self.multiTracker.add(self.tracker, frame, box)
                (x, y, w, h) = box
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                self.center_points_cur_frame.append((cx, cy))
                #print("FRAME N°", count, " ", x, y, w, h)

                # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else: 
            print("MOT")
            boxes = []
            success, boxes = multiTracker.update(frame)
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        

        # Only at the beginning we compare previous and current frame
        if count <= 5:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    if distance < 20:
                        tracking_objects[self.track_id] = pt
                        track_id += 1
        else:

            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = self.center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # Update IDs position
                    if distance < 20:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in self.center_points_cur_frame:
                            self.center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        print("Tracking objects")
        print(tracking_objects)


        print("CUR FRAME LEFT PTS")
        print(center_points_cur_frame)

        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()


    #When there is a new image updated, call this function
    def callback(self, image, depth, camera_info):
        if(self.frame > 3600):
            self.frame=0
        self.frame += 1     

        try:
            #convert ros compressed image message to cv2 image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(image)
            cv_depth = self.bridge.imgmsg_to_cv2(depth)

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

        #tracking target
        object_tracker(cv_image)
        #publish detection result and performance
        self.output_pub.publish(str(self.boxes)+'|'+str(self.confidences)+'|'+str(self.class_ids))      
        self.fps_pub.publish(str("{:.2f}".format(self.avg_fps)))

        #filtering detection result
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        target_detected = False
        if len(indices) > 0:
            for i in indices.flatten():
                x, y = self.centers[i]

                angle_x, angle_y = self.getAngle(x, y)
                result = getCoordinates(y, x, cv_depth, camera_info)
                distance = result[2]/1000#float('{0:.3g}'.format(cv_depth[y][x]/1000))
                self.target_coordinates.linear.x, self.target_coordinates.linear.y = getCoordinate(y,x, distance)
                self.target_coordinates.linear.z = 0
                self.target_coordinates_pub.publish(self.target_coordinates)
                self.message_pub.publish("Target base on UAV coordinates:\n"+str("x: {:.3f}\n".format(self.target_coordinates.linear.x))+str("y: {:.3f}\n".format(self.target_coordinates.linear.y)))
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