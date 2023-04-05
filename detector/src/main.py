#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import numpy as np
import time
import cv2
import yaml


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)


def plot_bboxes(image, boxes, labels=None, colors=None, score=True, conf=None):
    # Define COCO Labels
    if labels is None:
        print("ERROR: Missing model configuration")
        return
    # Define colors
    if colors is None:
        colors = [(89, 161, 197), (67, 161, 255), (19, 222, 24), (186, 55, 2), (167, 146, 11), (190, 76, 98),
                  (130, 172, 179), (115, 209, 128), (204, 79, 135), (136, 126, 185), (209, 213, 45), (44, 52, 10),
                  (101, 158, 121), (179, 124, 12), (25, 33, 189), (45, 115, 11), (73, 197, 184), (62, 225, 221),
                  (32, 46, 52), (20, 165, 16), (54, 15, 57), (12, 150, 9), (10, 46, 99), (94, 89, 46), (48, 37, 106),
                  (42, 10, 96), (7, 164, 128), (98, 213, 120), (40, 5, 219), (54, 25, 150), (251, 74, 172),
                  (0, 236, 196), (21, 104, 190), (226, 74, 232), (120, 67, 25), (191, 106, 197), (8, 15, 134),
                  (21, 2, 1), (142, 63, 109), (133, 148, 146), (187, 77, 253), (155, 22, 122), (218, 130, 77),
                  (164, 102, 79), (43, 152, 125), (185, 124, 151), (95, 159, 238), (128, 89, 85), (228, 6, 60),
                  (6, 41, 210), (11, 1, 133), (30, 96, 58), (230, 136, 109), (126, 45, 174), (164, 63, 165),
                  (32, 111, 29), (232, 40, 70), (55, 31, 198), (148, 211, 129), (10, 186, 211), (181, 201, 94),
                  (55, 35, 92), (129, 140, 233), (70, 250, 116), (61, 209, 152), (216, 21, 138), (100, 0, 176),
                  (3, 42, 70), (151, 13, 44), (216, 102, 88), (125, 216, 93), (171, 236, 47), (253, 127, 103),
                  (205, 137, 244), (193, 137, 224), (36, 152, 214), (17, 50, 238), (154, 165, 67), (114, 129, 60),
                  (119, 24, 48), (73, 8, 110)]

    # plot each boxes
    for box in boxes:
        # add score in label if score=True
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1])]
        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    return image


class Detector:
    def __init__(self, config_file=None):
        self.cv_bridge = CvBridge()

        model_name = rospy.get_param('~yolo_model', "$(find detector)/model/yolov8n.pt")
        model_config_file = rospy.get_param('~model_config', "$(find detector)/config/yolov8.yaml")
        with open(model_config_file, "r") as file:
            self.model_config = yaml.load(file, Loader=yaml.Loader)
        # print(self.model_config)
        self.model = YOLO(model_name)

        rospy.Subscriber('/sensor_stack/cameras/stereo_front/zed_node/left/image_rect_color', Image,
                         self.get_image_cb)

        self.image_publisher = rospy.Publisher(
            '/detector_image',
            Image, queue_size=10)

        self.labels_publisher = rospy.Publisher(
            '/detector_classes',
            Float64MultiArray, queue_size=10
        )

    def get_image_cb(self, msg):
        start = time.time()
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return
        mid = time.time() - start
        # print(mid)

        yolo_results = self.model.predict(source=cv_image[:, :, ::-1], stream=True)

        for result in yolo_results:
            # print(result.boxes.boxes)
            my_msg = Float64MultiArray() 
            d = result.boxes.boxes.tolist()
            my_msg.layout.dim.append(MultiArrayDimension())
            my_msg.layout.dim.append(MultiArrayDimension())

            width = 6
            height = len(d)
            my_msg.layout.dim[0].size = height
            my_msg.layout.dim[1].size = width
            my_msg.layout.dim[0].label = "height"
            my_msg.layout.dim[1].label = "width"
            my_msg.layout.dim[0].size = height
            my_msg.layout.dim[1].size = width
            my_msg.layout.dim[0].stride = width*height
            my_msg.layout.dim[1].stride = width
            my_msg.layout.data_offset = 0

            # d = [[float(d[i][j]) for j in range(len(d[0]))] for i in range(len(d))]
            # print([[0] * 6] * len(d))
            my_msg.data = [item for sublist in d for item in sublist]

            cv_image_box = plot_bboxes(cv_image, result.boxes.boxes, labels=self.model_config['labels'], conf=0.5)
            mid = time.time() - mid
            # print(mid)
            cv_image_box = self.cv_bridge.cv2_to_imgmsg(cv_image_box, encoding="bgr8")
            # print(time.time() - mid)
            self.image_publisher.publish(cv_image_box)
            self.labels_publisher.publish(my_msg)


if __name__ == '__main__':
    rospy.init_node('Detector', anonymous=True)
    model = Detector()
    rospy.spin()
