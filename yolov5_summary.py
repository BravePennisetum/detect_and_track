# import the libraries
from __future__ import print_function

from tracking.tracking_constants import *
from detection.yolov5.yolov5 import YoloV5
from utils.messages import initial_prints
from utils.general import is_video_file, is_video_stream, Dirs, setup_video_writer, expand_bboxes
from tracking.tracking_tasks import track2
from time import strftime as time_strftime
import argparse
import os
import sys

if __name__ == '__main__':

    sys.path.insert(0, './detection/yolov5')

    """ Part 1 - Parse Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5_resize', type=int, default=640, help='Image Re-sizing by YOLOv5')
    parser.add_argument('--yolov5_weights', type=str, default='./detection/yolov5/weights/yolov5s_cxs.pt', help='Weights of yolov5 by YOLOv5')
    opt = parser.parse_args()

    """ Part2 - Setup constants and CV objects"""
    # yolo resize
    yolo_resize = opt.yolov5_resize
    # yolo weights
    yolo_weights = opt.yolov5_weights

    # Create a YoloV5 object
    yolov5 = YoloV5(weights=yolo_weights, device='cpu', img_size=yolo_resize)

