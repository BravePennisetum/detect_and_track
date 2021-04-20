import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from tracking.deep_sort.utils.draw import draw_boxes
from tracking.deep_sort.utils.io import write_results
from tracking.deep_sort.utils.parser import get_config
from detection.yolov5.models.experimental import attempt_load
from detection.yolov5.utils.datasets import LoadStreams, LoadImages
from detection.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, \
    increment_path, strip_optimizer, check_imshow
from detection.yolov5.utils.torch_utils import select_device, time_synchronized

from tracking.deep_sort.deep_sort import build_tracker

from tracking.tracking_constants import *

import sys
import argparse
import traceback
""" YOLOv5 - 19/4/2021 """

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']


# TODO - to be removed in utils file
def names_to_indexes(class_names, class_list):
    class_idxs = []
    for class_name in class_names:
        for i, cls in enumerate(class_list):
            if class_name == cls:
                class_idxs.append(i)
    return class_idxs


# All is run in one function to avoid the delays from calling a function and returning etc...
def detect_and_track4(opt):
    # If you find any errors when loading YOLOv5 try removing uncommenting the line below and try again.
    # sys.path.insert(0, '../detection/yolov5/weights')

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # save deep sort results
    save_results_path = os.path.join(save_dir, "deep-sort_results.txt")
    # Deep SORT configurations
    if 'original' in os.path.split(opt.config_deepsort)[1]:
        use_original_deep_sort = True
    else:
        use_original_deep_sort = False
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    """ YOLOv5 """
    # Load Detector model
    detector_model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(detector_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        detector_model.half()  # to FP16

    """ Deep SORT """
    # Set up Deep Sort Tracker
    # Load Tracker Model
    deepsort_config = get_config()
    deepsort_config.merge_from_file(opt.config_deepsort)
    if opt.device != 'cpu':
        deepsort = build_tracker(deepsort_config, use_cuda=True, use_original_deep_sort=use_original_deep_sort)
    else:
        deepsort = build_tracker(deepsort_config, use_cuda=False, use_original_deep_sort=use_original_deep_sort)

    # Set Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # set up video_path
    save_video_path = os.path.join(save_dir, 'test_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_video_path, fourcc, dataset.fps,
                                   (dataset.input_frame_size[0], dataset.input_frame_size[1]))

    # Get names and colors
    names = detector_model.module.names if hasattr(detector_model, 'module') else detector_model.names
    class_list = names
    print('\n- Available classes for detection:\n', names)
    colors_db = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        detector_model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(detector_model.parameters())))  # run once
    t0 = time.time()

    """ -- Manos Addition -- """
    bboxes = []
    colours = []
    classes = []
    frame_number = 0
    results = []
    try:
        for path, img, im0s, vid_cap in dataset:
            # Original imaage
            frame = im0s
            frame_number += 1

            """ DETECTION by YOLOv5 """
            # img ==> transformed image for yolov5
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = detector_model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            # Initialize bboxes
            bboxes = []
            colours = []
            classes = []
            cls_conf = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame_counter_from_dataset_object = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame_counter_from_dataset_object = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for xmin, ymin, xmax, ymax, conf, cls in det.tolist():
                        w = xmax - xmin
                        h = ymax - ymin
                        # Add a bbox
                        # Deep sort will take bboxes as ['x_center', 'y_center', 'w', 'h']
                        bboxes.append([xmin + w / 2, ymin + h / 2, w, h])
                        # Add current box's color
                        colours.append(colors_db[int(cls)])
                        # Add current box's class
                        classes.append(names[int(cls)])
                        # Add current box's class confidence
                        cls_conf.append(conf)

            print(f'○ YOLOv5 frame process done in "{time.time() - t1:.3f}" seconds.')

            """ TRACKING by deep sort"""

            # Deep Sort is already initialized
            if len(bboxes) > 0:
                bboxes_tensor = torch.FloatTensor(bboxes)

                class_indexes = names_to_indexes(classes, class_list)
                classes_tensor = torch.LongTensor(class_indexes)
                cls_conf_tensor = torch.FloatTensor(cls_conf)
            else:
                bboxes_tensor = torch.FloatTensor([]).reshape([0, 4])
                cls_conf_tensor = torch.FloatTensor([])
                classes_tensor = torch.LongTensor([])

            # track objects of 'boat' class
            # mask = classes_tensor == 8
            mask = torch.BoolTensor([True for _ in range(len(bboxes))])
            bbox_xywh = bboxes_tensor[mask]

            # expand boxes - this line can be removed
            bbox_xywh[:, 3:] *= 1.2

            # get class confidences
            cls_conf_to_use = cls_conf_tensor[mask]

            # time point to measure deep SORT update duration
            start_deep_sort = time.time()

            # do tracking
            outputs, cls_names = deepsort.update(bbox_xywh, cls_conf_to_use, frame, classes_tensor)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                class_names = [class_list[cls_name] if cls_name != -1 else "" for cls_name in cls_names]
                frame = draw_boxes(frame, bbox_xyxy, identities, class_names=class_names)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(deepsort.xyxy_to_tlwh(bb_xyxy))

                results.append((frame_number - 1, bbox_tlwh, identities))

            # save results
            write_results(save_results_path, results, 'mot')
            print(f'♦ Deep SORT frame process lasted "{time.time() - start_deep_sort:.3f}" seconds.',
                  '\n--------------------------------------------------------------')

            # End of pipeline
            waitKey = cv2.waitKey(delay_value)
            if waitKey & 0xFF == 27:
                print('\n- Button Pressed: "Esc".\n')
                break
            elif waitKey & 0xFF == ord('q'):
                print('\n- Button Pressed: "q".\n')
                break
            else:
                cv2.imshow('YOLOv5 x Deep SORT', frame)
                video_writer.write(frame)
                continue
    except Exception as e:
        traceback.print_exc()

    print('Ending detection and tracking. Exiting...')
    video_writer.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/seagull_50.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='videos/sample2.mp4', help='source')  # file/folder, 0 for webcam
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
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    opt_ = parser.parse_args()
    print(opt_)

    with torch.no_grad():
        if opt_.update:  # update all models (to fix SourceChangeWarning)
            for opt_.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect_and_track4(opt_)
                strip_optimizer(opt_.weights)
        else:
            detect_and_track4(opt_)
