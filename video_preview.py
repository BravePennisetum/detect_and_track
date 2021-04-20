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
    parser.add_argument('--source', default='videos/sample2.mp4',
                        help='Use "videos/sample2.mp4" by default. Give a file\'s path.')
    opt = parser.parse_args()

    # define inputs path
    videoPath = opt.source

    # Extract video type
    video_format = videoPath[-3:]

    ## warnings.filterwarnings('ignore')  # to disable warnings

    # Date and time of starting point
    starting_time = time_strftime("%H-%M-%S - %d.%m.%Y")

    # Check if input is correct.
    # TODO - This part needs to support streaming.
    if (not is_video_file(videoPath)) and (not is_video_stream(videoPath)):
        raise Exception(
            "Using a wrong input format '{video_format}'. Please use 'AVI', 'MP4' for video files,\n or 'RTSP', 'HTTP', 'RTMP' for video streams!".format(
                video_format=videoPath[-3:]))

    # Initialize some important directory variables
    video_dir, video_name = os.path.split(videoPath)
    print(videoPath)
    # Initialize VideoCapture object
    cam = cv2.VideoCapture(videoPath)
    # Initialize VideoWriter
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))

    # Print the initial Messages
    initial_prints(starting_time, fps)

    # Initialize other variables
    current_frame_number = -1

    """ Part3 - Process Video """
    while cam.isOpened():
        # Read one frame
        success, frame = cam.read()
        if not success:
            print("Video reached the end...")
            break

        current_frame_number += 1
        print('Frame:', current_frame_number)

        cv2.imshow('Detector__MultiTracker', frame)
        cv2.waitKey(int(1000/fps))

    cam.release()
    cv2.destroyAllWindows()
