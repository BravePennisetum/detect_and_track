# import the libraries
from __future__ import print_function

from PIL import Image

from tracking.tracking_constants import *
from utils.messages import initial_prints
from utils.general import is_video_file, is_video_stream, Dirs
from tracking.tracking_tasks import track
from time import strftime as time_strftime
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', type=str, default='CSRT',
                        help='Use one of the following: ["CSRT","MIL","KCF","TLD","MedianFlow", "GOTURN", "MOSSE","Boosting"]')
    parser.add_argument('--results_dir', default='tracked/', help='save results to tracked/')
    parser.add_argument('--save_type', default='video', help='Save results as "image"s or "video".')
    parser.add_argument('--source', default='videos/sample2.mp4',
                        help='Use "videos/sample2.mp4" by default. Give a file\'s path.')
    parser.add_argument('--crop_images', type=bool, default=False)
    parser.add_argument('--crop_directory', type=str, default='tracked/cropped/')
    opt = parser.parse_args()
    # print("Available tracking algorithms are:\n")
    # for t in trackerTypes:
    #    print(t)

    # create tracker type
    trackerType = opt.tracker
    # define results_dir
    tracked_dir = opt.results_dir
    # define inputs path
    videoPath = opt.source
    # define type of file to save - images or videos
    dataset_mode = opt.save_type
    # crop images ????
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

    video_dir, video_name = os.path.split(videoPath)

    dirs = Dirs(video_name, tracked_dir, starting_time, trackerType)

    cam = cv2.VideoCapture(videoPath)
    # fps = cam.get(30)  # OR SET IT MANUALLY
    fps = cam.get(cv2.CAP_PROP_FPS)

    """ Prints the initial messages """
    initial_prints(starting_time, fps)

    """ Extract an image frame from the video as image """
    # frame
    currentframe = 0  # set the frame

    # reading from frame
    ret, frame = cam.read()

    # if video is still left continue creating images
    name = dirs.get_starting_frame_dir() + './starting_frame.jpg'

    # writing the extracted images
    cv2.imwrite(name, frame)

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    # *********************************************************************************************************************************************************************************************************************************************

    # *************************************** MOT *********************************************************************************************************************************************************************************************************
    # filename = "image_framesstarting_frame_0.jpg"
    with Image.open(name) as image:
        width, height = image.size

    track(
        trackerType,
        videoPath,
        video_format,
        dataset_mode,
        dirs,
        starting_time,
        i_should_crop_images,
        crop_dir
    )
