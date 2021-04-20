import os
import cv2
import numpy as np
from pathlib import Path


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def create_dirs(video_name: str, tracked_dir, starting_time: str, trackerType: str):
    if not tracked_dir in os.getcwd():
        tracked_dir = os.path.join(os.getcwd(), tracked_dir)

    dir_starting_time = os.path.join(tracked_dir, video_name + ' - ' + starting_time + ' - ' + trackerType)
    os.makedirs(dir_starting_time)

    dir_starting_frame = os.path.join(dir_starting_time, 'starting_frame')
    os.makedirs(dir_starting_frame)

    dir_command = os.path.join(dir_starting_time, 'event_information')
    os.makedirs(dir_command)

    dir_extracted_video = os.path.join(dir_starting_time, 'video_information')
    os.makedirs(dir_extracted_video)

    dir_streaming_frames = os.path.join(dir_starting_time, 'streaming')
    os.makedirs(dir_streaming_frames)

    out_path = os.path.join(dir_starting_time, 'video.mp4')

    # dir_crop = os.path.join(dir_starting_time, 'crop')
    # os.makedirs(dir_crop)

    # dir_geoloc = os.path.join(dir_starting_time, 'geolocations')
    # os.makedirs(dir_geoloc)

    # outputvideoPath = dir_extracted_video+'./tracking_'+starting_time+'.mp4'
    # outputvideoPath_streaming = dir_extracted_video+'./streaming_'+starting_time+'.mp4'

    return dir_starting_time, dir_starting_frame, dir_command, dir_extracted_video, dir_streaming_frames, out_path


class Dirs:
    def __init__(self, video_name: str, tracked_dir, starting_time: str, trackerType: str):
        if not tracked_dir in os.getcwd():
            tracked_dir = os.path.join(os.getcwd(), tracked_dir)

        self.dir_starting_time = os.path.join(tracked_dir, video_name + ' - ' + starting_time + ' - ' + trackerType)
        os.makedirs(self.dir_starting_time)

        self.dir_starting_frame = os.path.join(self.dir_starting_time, 'starting_frame')
        os.makedirs(self.dir_starting_frame)

        self.dir_command = os.path.join(self.dir_starting_time, 'event_information')
        os.makedirs(self.dir_command)

        self.dir_extracted_video = os.path.join(self.dir_starting_time, 'video_information')
        os.makedirs(self.dir_extracted_video)

        self.dir_streaming_frames = os.path.join(self.dir_starting_time, 'streaming')
        os.makedirs(self.dir_streaming_frames)

        self.out_path = os.path.join(self.dir_starting_time, 'video.mp4')

    # TODO - find better names for directories
    def get_starting_time_dir(self):
        return self.dir_starting_time

    def get_starting_frame_dir(self):
        return self.dir_starting_frame

    def get_command_dir(self):
        return self.dir_command

    def get_extracted_video_dir(self):
        return self.dir_extracted_video

    def get_streaming_frames_dir(self):
        return self.dir_streaming_frames

    def get_output_path(self):
        return self.out_path

    def get_out_path(self):
        return self.out_path


def expand_bboxes(bboxes, frame_width, frame_height, c=10):
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        # Top left corner
        if x - c / 2 >= 0:
            bbox[0] = int(x - c / 2)
        else:
            bbox[0] = 0
        if y - c / 2 >= 0:
            bbox[1] = int(bbox[1] - c / 2)
        else:
            bbox[1] = 0

        # Width & height
        if x + w + c / 2 < frame_width:
            bbox[2] = int(w + c)
        else:
            d = frame_width - x
            bbox[2] = int(d)

        if y + h + c / 2 < frame_height:
            bbox[3] = int(h + c)
        else:
            d = frame_height - y
            bbox[3] = int(d)
    return bboxes


def write_to_txt(stream, message):
    stream.write(message + '\n')


def write_list_to_txt(stream, message_list):
    for message in message_list:
        stream.write(message + '\n')


def is_video_file(video_path: str):
    print(video_path.endswith('mp4'))
    return video_path.endswith('mp4') or video_path.endswith('avi')


def is_video_stream(video_path: str):
    return video_path.isnumeric() or video_path.lower().startswith(('rtsp://', 'rtmp://', 'http://'))


def check_imshow():
    # Check if environment supports image displays
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False


def setup_video_writer(output_path, width, height, fps, video_format):
    # Create a video capture object to read videos

    if video_format == 'avi' or video_format == 'AVI':
        print('Choosing XVID codec')
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    elif video_format == 'mp4' or video_format == 'MP4':
        print('Choosing MP4V codec')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # Create video writer
    else:
        raise Exception("tracking_tasks Error (track2): Not supported video format '{video_format}'".format(
            video_format=video_format))
    return cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


def skip_frames(cam):
    while 1:
        success, frame = cam.read()
        cv2.imshow('MultiTracker', frame)
        if not success:
            break
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break
        if ord('m') == k:
            continue
