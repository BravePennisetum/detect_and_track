import argparse
import torch
from detect_and_track.yolov5_x_deep_sort import detect_and_track4
from detection.yolov5.utils.general import strip_optimizer
import sys

if __name__ == '__main__':

    sys.path.insert(0, 'detection/yolov5')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov5s_cocoXseagull_50.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../tracking data/videos/uav_rgb.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="./tracking/deep_sort/configs/original_deep_sort.yaml",
                        help='configuration file for deep SORT initialization.')
    opt_ = parser.parse_args()
    print(opt_)

    with torch.no_grad():
        if opt_.update:  # update all models (to fix SourceChangeWarning)
            for opt_.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect_and_track4(opt_)
                strip_optimizer(opt_.weights)
        else:
            detect_and_track4(opt_)
