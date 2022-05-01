#!/usr/bin/env python3
from __future__ import print_function

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

#set name of rosnode
rospy.init_node('user_control', anonymous=True)

#mask detection output format
fmt = "{:10}{:15}{:15}{:20}"
pi=3.14159
#show image and draw detection result 
class boxes_drawer:
    def __init__(self):
        self.bridge = CvBridge()
        
        #collect detected result
        self.boxes = []
        self.classes = ["with mask", "without mask", "test"]
        self.class_ids = []
        self.confidences = []
        self.indices = []
        self.cv_image = []

        #boxes color
        self.colorizer = rs.colorizer()

        #alert period
        self.last_time = time.time()

        #display detection performance(show fps)
        self.fps = "0"


    def start(self):
        #get image from camera
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.getImage)

        #get detection result
        self.boxes_sub = rospy.Subscriber("/human_tracking/mask_detection/boxes", String, self.getBoxes)

        #get detection fps
        self.fps_sub = rospy.Subscriber("/human_tracking/mask_detection/fps", String, self.getFPS)

        #get detection fps
        self.message_sub = rospy.Subscriber("/user_control/message", String, self.getMessage)

        #listen alert event
        self.sound_sub = rospy.Subscriber("/user_control/play_sound", String, self.playSound)

        #publish alert event
        self.sound_pub = rospy.Publisher("/user_control/play_sound", String, queue_size=10)

    def getFPS(self, msg):
        self.fps = msg.data

    def getMessage(self, msg):
        mission_state_label["text"] = msg.data
        pass

    def playSound(self, msg):
        #play alert sound effect
        playsound("/home/ho/Downloads/beep-beep-6151.mp3")

    def stop(self):
        try:
            #unregiste all topic
            self.image_sub.unregister()
            self.boxes_sub.unregister()

            #reset detection result
            self.boxes = []
            self.confidences = []
            self.indices=[]
        except: #if subsriber didnt registered, pass is ok
            pass

    def getImage(self, image):
        try:
            #change ros image msg to cv2
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(image)
        except CvBridgeError as e:
            print(e)

        #show image on GUI
        self.showImage(self.cv_image)

    def showImage(self, cv_image, windowName = "output"):
        #box color: blue
        color = (255, 1, 12)

        #update detection result GUI
        result_listbox.delete(1, tk.END)

        #counter for mask and people detection result
        without_mask = 0
        people = 0

        #checking for new detection result
        if len(self.indices) > 0:
            #update detector status
            detector_status_label["text"]="Started"
            detector_status_label["bg"]="green"

            #show each result
            for i in self.indices.flatten():
                try:
                    #get coordinates
                    (x, y) = (self.boxes[i][0], self.boxes[i][1])
                    (w, h) = (self.boxes[i][2], self.boxes[i][3])

                    #darw bounding box                    
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)     

                    #label content
                    text = "{}: {:.2f}".format(self.classes[self.class_ids[i]], self.confidences[i])

                    #put label
                    cv2.rectangle(cv_image, (x, y-28), (x + w, y), (255,255,255), -1)            
                    cv2.putText(cv_image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, lineType=cv2.LINE_AA)
                    
                    #show detection result to GUI
                    result_listbox.insert(tk.END, fmt.format(str(i),str(x),str(y),str(self.classes[self.class_ids[i]])))
                    if(self.class_ids[i] == 1):
                        without_mask +=1
                    people+=1
                    
                except:
                    continue   
        
        #check mask detection result
        if(without_mask==0):
            number_of_without_mask["text"]="0 people without mask"
            number_of_without_mask["bg"]="white"
        else:
            number_of_without_mask["text"]=str(without_mask)+" people without mask"
            number_of_without_mask["bg"]="red"
            current_time = time.time()

            #check alert period
            if((current_time - self.last_time) >= 1):
                self.last_time = current_time
                self.sound_pub.publish("play")
                print("play sound")

        number_of_people["text"]=str(people)        

        #change cv2 image to tkinter image 
        b, g, r = cv2.split(cv_image)
        img = cv2.merge((r,g,b))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        #show image
        image_label.configure(image=imgtk)
        image_label.imgtk = imgtk

    #get detection result
    def getBoxes(self, boxes):
        data = boxes.data
        boxes, confidences, class_ids = data.split("|")
        
        self.boxes = literal_eval(boxes)
        self.confidences = literal_eval(confidences)
        self.class_ids = literal_eval(class_ids)
        self.indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)

        FPS_label["text"] = self.fps

#flight control and monitoring mavros state
class flight_control:
    def __init__(self):
        #mavros state
        self.isConnected = False
        self.isArmed = False
        self.flight_mode = "manual"

        self.current_angle = -1.00
        self.current_x = 0.00
        self.current_y = 0.00

        #get mavros state
        self.mavros_state_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state)

        #get current state
        self.current_pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.current_pose)

        #get target position
        self.target_sub = rospy.Subscriber("/human_tracking/mask_detection/target", Twist, self.getTarget)

        #set flight mode for offboard mode program (offb_mode.cpp)
        self.mavros_mode_pub = rospy.Publisher("/user_control/set_mode", String, queue_size=10)

        self.target_coordinates_sub = rospy.Subscriber("/human_tracking/mask_detection/target/coordinates", Twist, self.coordinates_transfrom)

    def coordinates_transfrom(self, msg):
        target_x = msg.linear.x
        target_y = msg.linear.y
        transformed_x = self.current_x + target_x*np.cos((self.current_angle)*(pi/180))-target_y*np.sin((self.current_angle)*(pi/180))
        transformed_y = self.current_y + target_x*np.sin((self.current_angle)*(pi/180))+target_y*np.cos((self.current_angle)*(pi/180))
        target_x_label["text"]="{:.12f}".format(transformed_x)
        target_y_label["text"]="{:.12f}".format(transformed_y)

    def getTarget(self, msg):
        #update to GUI
        #target_x_label["text"]="{:.12f}".format(msg.linear.x)
        #target_y_label["text"]="{:.12f}".format(msg.linear.y)
        #target_z_label["text"]="{:.12f}".format(msg.linear.z)
        pass



    def mavros_state(self, state):
        self.isConnected = state.connected
        if( self.isConnected==True):
            mavros_label["text"]="connected"
            mavros_label["bg"]="green"
        else:
            mavros_label["text"]="disconnected"
            mavros_label["bg"]="red"
        
        #if uav armed, enable offboard, position or start task. otherwise, disable all function
        self.isArmed = state.armed
        if(self.isArmed==True):
            mode_arm_button["bg"]="green"
            mode_land_button["state"] = "normal"
            mode_offboard_button["state"] = "normal"
            mode_position_button["state"] = "normal"
            mode_task_button["state"] = "normal"
        else:
            mode_arm_button["bg"]="red"
            mode_land_button["state"] = "disabled"
            mode_offboard_button["state"] = "disabled"
            mode_position_button["state"] = "disabled"
            mode_task_button["state"] = "disabled"

        self.flight_mode = state.mode

    def current_pose(self, msg):
        #update to GUI
        current_x_label["text"] = "{:.12f}".format(msg.pose.position.x)
        current_y_label["text"] = "{:.12f}".format(msg.pose.position.y)
        current_z_label["text"] = "{:.12f}".format(msg.pose.position.z)
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        if(z>0 and w>0):
            self.current_angle =(np.arcsin(z)+np.arccos(w))*(180/pi)
            current_orientation_label["text"]="{:.2f}".format(self.current_angle)
        elif(z>0 and w<0):
            self.current_angle =(np.arccos(z)+np.arccos(w))*(180/pi)+90
            current_orientation_label["text"]="{:.2f}".format(self.current_angle)
        elif(z<0 and w<0):
            self.current_angle =(np.arccos(z)+np.arcsin(w))*(180/pi)
            current_orientation_label["text"]="{:.2f}".format(self.current_angle)
        elif(z<1 and w>0):
            self.current_angle =450-(np.arccos(z)+np.arccos(w))*(180/pi)
            current_orientation_label["text"]="{:.2f}".format(self.current_angle)

       

    def set_mode(self, mode):
        #publish mode to offboard mode
        self.mavros_mode_pub.publish(mode)


#camera state button event
def setCameraState():
    if(camera_state_button["text"]=="Turn on camera"):
        camera_state_button["text"]="Turn off camera"
        print("Turn on camera")
        image_label["text"] = "No image message published"
        boxDrawer.start()

    elif(camera_state_button["text"]=="Turn off camera"):
        camera_state_button["text"]="Turn on camera"
        print("Turn off camera")
        image_label["text"] = "UAV camera disabled"
        boxDrawer.stop()
        image_label.configure(image='')

#mask detection button event
def setMaskDetectionState():
    if(mask_detect_button["text"]=="Start mask detection"):
        #update button text
        mask_detect_button["text"]="Stop mask detection"

        #update detector status
        detector_status_label["text"]="Loading..."
        detector_status_label["bg"]="yellow"

        #resize window
        root.geometry('1330x720')
        print("Start mask detection")

        os.system("gnome-terminal -- roslaunch human_tracking mask_detection.launch")


    elif(mask_detect_button["text"]=="Stop mask detection"):
        #update button text
        mask_detect_button["text"]="Start mask detection"

        #update detector status
        detector_status_label["text"]="Disabled"
        detector_status_label["bg"]="red"

        #resize window
        root.geometry('1000x720')
        print("Stop mask detection")
        
        os.system("rosnode kill mask_detection")

#flight mode button event
def setFlightMode(mode = 0):
    if(mode==0):
        if(flightControl.isArmed==True):
            os.system("rosnode kill offb_node")
        else:
            os.system("gnome-terminal -- roslaunch human_tracking human_tracking.launch")

    elif(mode == 1):
        flightControl.set_mode("onboard")
    elif(mode == 2):
        print("Flight mode set to Position mode")
    elif(mode == 3):
        flightControl.set_mode("task1")
    elif(mode == 3):
        print("Uav landing")

#GUI design##################################

#Label#######################################
root = tk.Tk()
root.title("User Control")
root.geometry('1000x720')

label_font = tkFont.Font(family='Times', size=22)
title_label = tk.Label(root)
title_label["font"] = label_font
title_label["text"] = "User Control GUI"
title_label["justify"] = "center"
title_label.place(x=360, y=15)

label_font = tkFont.Font(family='Times', size=18)
status_lable = tk.Label(root)
status_lable["font"] = label_font
status_lable["text"] = "UAV status"
status_lable.place(x=700, y=60)

label_font = tkFont.Font(family='Times', size=18)
image_label = tk.Label(root)
image_label["font"] = label_font
image_label["text"] = "UAV camera disabled"
image_label["bg"] = "white"
image_label.place(x=10, y=60, height=480, width=640)

label = tk.Label(root, text="Mavros connection:")
label.place(x=680, y=100)

mavros_label = tk.Label(root, text="disconnected")
mavros_label["bg"] = "red"
mavros_label.place(x=820, y=100)

#set flight mode####################################
flight_mode_frame = tk.LabelFrame(root, text="Flight Mode")
flight_mode_frame.place(x=680, y=130, height=130, width=300)

mode_arm_button = tk.Button(flight_mode_frame, text="Arm", command=lambda:setFlightMode(0),width=7, height=1)
mode_arm_button["bg"] = "red"
mode_arm_button.grid(row=0, column=0, padx=5, pady=2)


mode_offboard_button = tk.Button(flight_mode_frame, text="Offboard", command=lambda:setFlightMode(1),width=7, height=1)
mode_offboard_button["state"] = "disabled"
mode_offboard_button.grid(row=1, column=0, padx=5, pady=2)

mode_position_button = tk.Button(flight_mode_frame, text="Position", command=lambda:setFlightMode(2),width=7, height=1)
mode_position_button["state"] = "disabled"
mode_position_button.grid(row=1, column=1, padx=5, pady=2)

mode_task_button = tk.Button(flight_mode_frame, text="Start Task", command=lambda:setFlightMode(3),width=7, height=1)
mode_task_button["state"] = "disabled"
mode_task_button.grid(row=1, column=2, padx=5, pady=2)

mode_land_button = tk.Button(flight_mode_frame, text="Land", command=lambda:setFlightMode(4),width=7, height=1)
mode_land_button["state"] = "disabled"
mode_land_button.grid(row=2, column=0, padx=5, pady=2)
#####################################################

#show current position###############################
current_pose_frame=tk.LabelFrame(root, text="Current position")
current_pose_frame.place(x=680, y=280, height=90, width=300)

label = tk.Label(current_pose_frame, text="        X:    ")
label.grid(row=0, column=0)
label = tk.Label(current_pose_frame, text="        Y:    ")
label.grid(row=1, column=0)
label = tk.Label(current_pose_frame, text="        Z:    ")
label.grid(row=2, column=0)

current_x_label = tk.Label(current_pose_frame, text="0.000000000")
current_x_label["bg"]="white"
current_x_label["justify"]="left"
current_x_label.grid(row=0, column=1)
current_y_label = tk.Label(current_pose_frame, text="0.000000000")
current_y_label["bg"]="white"
current_y_label["justify"]="left"
current_y_label.grid(row=1, column=1)
current_z_label = tk.Label(current_pose_frame, text="0.000000000")
current_z_label["bg"]="white"
current_z_label["justify"]="left"
current_z_label.grid(row=2, column=1)

label = tk.Label(current_pose_frame, text="Orentation")
label["justify"]="center"
label.grid(row=0, column=2, padx=20)
current_orientation_label = tk.Label(current_pose_frame, text="-")
current_orientation_label["bg"]="white"
current_orientation_label["justify"]="center"
current_orientation_label.grid(row=1,column=2,padx=20)
#########################################################

#show current mission####################################
mission_frame=tk.LabelFrame(root, text="Message")
mission_frame.place(x=680, y=390, height=300, width=300)

mission_state_label=tk.Label(mission_frame, text="Land")
mission_state_label["bg"]="white"
mission_state_label.place(x=20, y=5)
#########################################################

#switch camera state
camera_state_button = tk.Button(root, text="Turn on camera", command=setCameraState)
camera_state_button.place(x=200, y=550)

#for mask detection control
mask_detect_button = tk.Button(root, text="Start mask detection", command=setMaskDetectionState)
mask_detect_button.place(x=340, y=550)

#mask detection status
label_font = tkFont.Font(family='Times', size=18)
label = tk.Label(root, text="Mask detection status")
label["font"] = label_font
label.place(x=1020, y=60)

label = tk.Label(root, text="Detector: ")
label.place(x=1000, y=100)

detector_status_label = tk.Label(root, text="Disabled")
detector_status_label["bg"]="red"
detector_status_label.place(x=1070, y=100)

#mask detect result
mask_detect_frame =tk.LabelFrame(root, text="Result")
mask_detect_frame.place(x=1000, y=130, height=200, width=300)

number_of_people = tk.Label(mask_detect_frame, text="0")
number_of_people.place(x=10, y=5)

label = tk.Label(mask_detect_frame, text="people detected")
label.place(x=25, y=5)

FPS_label = tk.Label(mask_detect_frame, text="-")
FPS_label.place(x=242, y=5)

label = tk.Label(mask_detect_frame, text="FPS:")
label.place(x=210, y=5)

result_listbox = tk.Listbox(mask_detect_frame)
result_listbox.insert(tk.END, fmt.format("ID", "X", "Y", "Class"))
result_listbox.place(x=8, y=30, height=100, width=280)

number_of_without_mask = tk.Label(mask_detect_frame, text="0 people without mask")
number_of_without_mask.place(x=10, y=140)

#track target detail
track_target_labelframe=tk.LabelFrame(root, text="Track target")
track_target_labelframe.place(x=1000, y=340, height=95, width=300)

label = tk.Label(track_target_labelframe, text="        X:    ")
label.grid(row=0, column=0)
label = tk.Label(track_target_labelframe, text="        Y:    ")
label.grid(row=1, column=0)
label = tk.Label(track_target_labelframe, text="        Z:    ")
label.grid(row=2, column=0)

target_x_label = tk.Label(track_target_labelframe, text="0")
target_x_label["bg"]="white"
target_x_label["justify"]="left"
target_x_label.grid(row=0, column=1, )
target_y_label = tk.Label(track_target_labelframe, text="0")
target_y_label["bg"]="white"
target_y_label["justify"]="left"
target_y_label.grid(row=1, column=1)
target_z_label = tk.Label(track_target_labelframe, text="0")
target_z_label["bg"]="white"
target_z_label["justify"]="left"
target_z_label.grid(row=2, column=1)


boxDrawer = boxes_drawer()

flightControl = flight_control()

#start program loop for callback and click event
root.mainloop()
