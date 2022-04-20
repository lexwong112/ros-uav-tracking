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
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

import rospy
import sys

import glob
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pyrealsense2 as rs
import message_filters

import tkinter as tk

class controlUI:
    def __init__(self):
        root = tk.Tk()
        root.title('Control')
        root.geometry('1280x720')
        task_btn = tk.Button(root, text='task off', command=lambda:self.control_task(1))
        task_btn.pack()

        task_btn = tk.Button(root, text='land', command=lambda:self.control_task(2))
        task_btn.pack()

        task_btn = tk.Button(root, text='task3', command=lambda:self.control_task(3))
        task_btn.pack()


        

        root.mainloop()

    def control_task(self, task):
        if(task == 1):
            os.system("gnome-terminal -x roslaunch human_tracking mask_detection.launch")       
        if(task == 2):
            print("task2")  
        if(task == 3):
            print("task3")  

def main(args):
    rospy.init_node('user_control', anonymous=True)
    control = controlUI()


if __name__ == '__main__':
    main(sys.argv)