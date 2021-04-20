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
    parser.add_argument('--tracker', type=str, default='CSRT',
                        help='Use one of the following: ["CSRT","MIL","KCF","TLD","MedianFlow", "GOTURN", "MOSSE","Boosting"]')
    parser.add_argument('--yolov5_resize', type=int, default=640, help='Image Re-sizing by YOLOv5')
    parser.add_argument('--yolov5_weights', type=str, default='./detection/yolov5/weights/yolov5s_cxs.pt', help='Weights of yolov5 by YOLOv5')
    # TODO - Change default to '0'
    parser.add_argument('--device', type=str, default='0',
                        help='Running inference on device of choice. To run on GPU pass the value "0"')
    parser.add_argument('--results_dir', default='tracked/', help='save results to tracked/')
    parser.add_argument('--save_type', default='video', help='Save results as "image"s or "video".')
    parser.add_argument('--source', default='videos/sample2.mp4',
                        help='Use "videos/sample2.mp4" by default. Give a file\'s path.')
    parser.add_argument('--crop_images', type=bool, default=False)
    parser.add_argument('--crop_directory', type=str, default='tracked/cropped/')
    parser.add_argument('--expand_box', type=int, default=40, help='Expand Bounding box by a constant factor.')
    opt = parser.parse_args()

    """ Part2 - Setup constants and CV objects"""
    # create tracker type
    trackerType = opt.tracker
    # yolo resize
    yolo_resize = opt.yolov5_resize
    # yolo weights
    yolo_weights = opt.yolov5_weights
    # Inference Device
    inference_device = opt.device
    # define results_dir
    tracked_dir = opt.results_dir
    # define inputs path
    videoPath = opt.source
    # define type of file to save - images or videos
    dataset_mode = opt.save_type
    # crop images ????
    i_should_crop_images = opt.crop_images
    crop_dir = opt.crop_directory
    # Box Expansion Constant
    c = opt.expand_box

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
    dirs = Dirs(video_name, tracked_dir, starting_time, trackerType)

    # Initialize VideoCapture object
    cam = cv2.VideoCapture(videoPath)
    # Initialize VideoWriter
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    video_writer = setup_video_writer(
        dirs.get_out_path(),
        frame_width,
        frame_height,
        fps,
        video_format
    )
    # fps = cam.get(30)  # OR SET IT MANUALLY
    fps = cam.get(cv2.CAP_PROP_FPS)

    # Print the initial Messages
    initial_prints(starting_time, fps)

    # Initialize other variables
    current_frame_number = -1

    # Create a YoloV5 object
    yolov5 = YoloV5(weights=yolo_weights, device=inference_device, img_size=yolo_resize)

    """ Part3 - Process Video """
    while cam.isOpened():

        # Read one frame
        success, frame = cam.read()
        if not success:
            print("Video reached the end...")
            break

        current_frame_number += 1

        # Run inference on YoloV5
        bboxes, colors = yolov5.detect(frame, current_frame_number)

        bboxes = expand_bboxes(bboxes, frame_width, frame_height, c=c)
        print('Bounding boxes detected:', bboxes)
        cv2.imshow('Detector x MultiTracker', frame)
        if len(bboxes) > 0:
            current_frame_number, end_task = track2(
                cvCapture=cam,
                frame=frame,
                current_frame_number=current_frame_number,
                bboxes=bboxes,
                colors=colors,
                trackerType=trackerType,
                dirs=dirs,
                video_writer=video_writer
            )
        else:
            # Continue until the detector makes a detection
            if cv2.waitKey(delay_value) & 0xFF == ord('q'):  # 'q' pressed
                break
            continue
        if end_task:
            break

    video_writer.release()
    cam.release()
    cv2.destroyAllWindows()
