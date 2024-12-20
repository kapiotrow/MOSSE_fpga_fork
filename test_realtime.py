#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import json
from deep_mosse import DeepMosse
import rospy
import cv2
from sensor_msgs.msg import Image
import cv2
from std_msgs.msg import Bool
import numpy as np
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge


my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))


class Cam(object):
    def __init__(self):
        self.frame = None
        self.tracker = None
        self.challenge_started = False
        self.dx = 0
        self.dy = 0
        self.tracker_output_pub = rospy.Publisher("/iris_control/mosse_output", Int32MultiArray, queue_size=10)
        self.bridge = CvBridge()
        self.fileno = 0
        self.config = None
        with open('configs/config.json', 'r') as json_file:
            self.config = json.load(json_file)
        rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.camera_sub_callback)
        rospy.Subscriber("/iris_control/challenge_start", Bool, self.start_challenge_cb)
        

    def camera_sub_callback(self, msg: Image):
        # img = np.array(list(msg.data)).astype(np.uint8)
        # self.frame = np.reshape(img, (msg.height, msg.width, 3))
        # frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if self.tracker==None:
            self.tracker = DeepMosse(self.frame, self.load_gt("groundtruth.txt")[0], config=self.config)
        else:
            position = self.tracker.track(self.frame)
            position = [round(x) for x in position]
            self.dx = int(100*(position[0]+(position[2]/2))/self.frame.shape[1])
            self.dy = int(100*(position[1]+(position[3]/2))/self.frame.shape[0])
            tracker_out = Int32MultiArray()
            tracker_out.data = [self.dx, self.dy]
            self.tracker_output_pub.publish(tracker_out)
            frame = self.frame.copy()
            #cv2.rectangle(self.frame, (x_min, y_min), (x_min + w, y_min + h), (255, 0, 0), 2, 1)
            cv2.rectangle(frame, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2, 1)
            cv2.imshow("tracked result", frame)
            #cv2.imwrite(my_absolute_dirpath + '/tank2/img_' + str(self.fileno) + '.png', self.frame)
            self.fileno+=1
            cv2.waitKey(1)


    def start_challenge_cb(self, msg):
        if msg.data:
            self.challenge_started = True


    def camera_sub(self):
        rospy.spin()


    def load_gt(self, gt_file, format='xyxy'):
        with open(gt_file, 'r') as file:
            lines = file.readlines()
        delimiters = [',', '\t']
        for d in delimiters:
            if d in lines[0]:
                lines = [line.split(d) for line in lines]
                break
        lines = [[int(float(coord)) for coord in line] for line in lines]
        result = [self.check_bbox(line) for line in lines]
        # returns in xywh format
        return result
    
    def check_bbox(self, points):
        Xs = points[::2]    # odd coords
        Ys = points[1::2]   # even coords
        left = min(Xs)
        right = max(Xs)
        top = min(Ys)
        bottom = max(Ys)
        width = right - left
        height = bottom - top
        return [left, top, width, height]

    


if __name__ == "__main__":
    rospy.init_node("camera_node_py")
    cam = Cam()
    rate = rospy.Rate(20)
    # print(cam.challenge_started)
    # while cam.challenge_started==False:
    #     rate.sleep()
    cam.camera_sub()
    cv2.destroyAllWindows()
    rate.sleep()