#!/usr/bin/env python3
from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from object_detection import ObjectDetection
import math
import copy
import rospy
import roslib
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('object_tracking', anonymous=True)

trackerTypes = ['KCF','TLD', 'MEDIANFLOW']

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  
  if trackerType == trackerTypes[0]:
    tracker = cv2.legacy.TrackerKCF_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.legacy.TrackerTLD_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.legacy.TrackerMedianFlow_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

if __name__ == '__main__':

  print("Default tracking algoritm is CSRT \n"
        "Available tracking algorithms are:\n")
  for t in trackerTypes:
      print(t)      

  trackerType = "KCF"     

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()
  
# Initialize Object Detection
od = ObjectDetection()

#cap = cv2.VideoCapture("los_angeles.mp4")

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0
boxes = []

bridge = CvBridge()

def callback(image):
    global count
    global bridge
    global track_id
    global center_points_prev_frame
    count += 1
    try:
        #convert ros compressed image message to cv2 image
        frame = bridge.compressed_imgmsg_to_cv2(image)


    except CvBridgeError as e:
        print(e)

    # Point current frame
    center_points_cur_frame = []
    if count%10 == 1:
        boxes = []
        (class_ids, scores, boxes) = od.detect(frame)
        bboxes = copy.copy(boxes)
        for box in bboxes:
            multiTracker.add(createTrackerByName(trackerType), frame, box)
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
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
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
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
  
    cv2.imshow("Frame", frame)
    cv2.waitKey(3)
    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()
rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, callback, queue_size=1)
rospy.Rate = 1/30
while not rospy.is_shutdown():
        rospy.sleep(rospy.Rate)

#cap.release()
cv2.destroyAllWindows()
