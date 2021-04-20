# import the libraries
from __future__ import print_function

from tracking.tracking_constants import *
from utils.messages import initial_prints
from utils.general import is_video_file, is_video_stream, Dirs, setup_video_writer
from utils.draw import draw_bboxes_with_mouse
from tracking.tracking_tasks import track2
from time import strftime as time_strftime
import argparse
import os

# TODO - Save the video as it is and the bboxes in a different file (preferably a CSV)
if __name__ == '__main__':
    """ Part 1 - Parse Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='CSRT',
                        help='Use one of the following: ["CSRT","MIL","KCF","TLD","MedianFlow", "GOTURN", "MOSSE","Boosting"]')
    parser.add_argument('--results_dir', default='tracked/', help='save results to tracked/')
    parser.add_argument('--save_type', default='video', help='Save results as "image"s or "video".')
    parser.add_argument('--source', default='videos/sample.avi',
                        help='Use "videos/sample.avi" by default. Give a file\'s path.')
    parser.add_argument('--crop_images', type=bool, default=False, help='Used to save images of the detected-tracked objects.')
    parser.add_argument('--crop_directory', type=str, default='tracked/cropped/')
    opt = parser.parse_args()

    """ Part2 - Setup constants and CV objects"""
    # create tracker type
    trackerType = opt.tracker
    # define results_dir
    tracked_dir = opt.results_dir
    # define inputs path
    videoPath = opt.source
    # define type of file to save - images or videos
    dataset_mode = opt.save_type
    # crop images
    i_should_crop_images = opt.crop_images
    crop_dir = opt.crop_directory

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
    video_writer = setup_video_writer(
        dirs.get_out_path(),
        int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cam.get(cv2.CAP_PROP_FPS)),
        video_format
    )
    # fps = cam.get(30)  # OR SET IT MANUALLY
    fps = cam.get(cv2.CAP_PROP_FPS)

    # Print the initial Messages
    initial_prints(starting_time, fps)

    # if video is still left continue creating images
    name = dirs.get_starting_frame_dir() + './starting_frame.jpg'

    # Release all space and windows once done
    # *****************************************************************************************************************
    # *************************************** MOT *********************************************************************

    ## Initialize other variables
    current_frame_number = -1
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("Video reached the end...")
            break
        bboxes, colors = draw_bboxes_with_mouse(frame)
        current_frame_number, end_task = track2(
            cvCapture=cam,
            frame=frame,
            current_frame_number=current_frame_number,
            bboxes=bboxes,
            colors=colors,
            trackerType=trackerType,
            dirs=dirs,
            starting_time=starting_time,
            video_writer=video_writer
        )

        if end_task:
            break

    video_writer.release()
    cam.release()
    cv2.destroyAllWindows()
