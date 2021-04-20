from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detection.yolov5.models.experimental import attempt_load
from detection.yolov5.utils.datasets import LoadStreams, LoadImages
from detection.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging, \
    increment_path, strip_optimizer, check_imshow
from detection.yolov5.utils.torch_utils import select_device, time_synchronized

from tracking.multi_tracker.tracker import createTrackerByName
from utils.general import expand_bboxes
from utils.draw import draw_bbox_text, draw_standard_text

from tracking.tracking_constants import *

import sys
import argparse

""" YOLOv5 - 5/3/2021 """


def detect_and_track3(opt):
    sys.path.insert(0, '../detection/yolov5/weights')

    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    """ MANOS ADDITION """
    trackerType = opt.trackerType
    c_expansion = opt.expand_box

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors_db = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    """ -- Manos Addition -- """
    detection_state = True
    tracking_state = False
    initiate_tracking = False
    multiTracker = None
    bboxes = []
    colours = []
    classes = []
    frame_number = 0
    frame_width, frame_height = dataset.input_frame_size
    for path, img, im0s, vid_cap in dataset:
        frame_number += 1
        """ DETECTION by YOLOv5 """
        if detection_state and not tracking_state:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Initialize bboxes
            bboxes = []
            colours = []
            classes = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for xmin, ymin, xmax, ymax, conf, cls in det.tolist():
                        w = xmax - xmin
                        h = ymax - ymin
                        # Add a bbox
                        bboxes.append([xmin, ymin, w, h])
                        colours.append(colors_db[int(cls)])
                        classes.append(names[int(cls)])

                    # After making a detection go to the trackers
                    detection_state = False
                    tracking_state = True
                    initiate_tracking = True

                # Stream results
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()

                    fourcc = 'mp4v'
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

            print(f'Detection Point: ({time.time() - t1:.3f}s)')

        elif not detection_state and tracking_state:
            """ TRACKING """
            if len(bboxes) <= 0:
                print('No objects detected. Entering Detection mode...')
                detection_state = True
                continue

            # Log everything
            """
            frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            """
            # The trackers work better if the bounding box is bigger than the object itself
            bboxes = expand_bboxes(bboxes, frame_width, frame_height, c=c_expansion)

            # Webcam frame comes as a list. The first element is the last frame (I presume). It also comes as a numpy array.
            # It is necessary to transform it to 'UMat'
            if webcam:
                frame = cv2.UMat(im0s[0])
            else:
                frame = im0s
            # Create MultiTracker object - recreate in order to re-enter new bboxes.
            if initiate_tracking or not multiTracker:
                multiTracker = cv2.MultiTracker_create()

                # Initialize MultiTracker - You can specify different trackers for every bounding box
                for bbox, color in zip(bboxes, colours):
                    rect = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    multiTracker.add(createTrackerByName(trackerType), frame, rect)

                initiate_tracking = False

            """ ------------- Start the main Loop ------------- """
            # get updated location of objects in subsequent frames
            t1 = time.time()
            success, boxes = multiTracker.update(frame)
            # if object is lost go to re-detect
            if not success:
                print('Lost Objects. Starting Detection...')
                detection_state = True
                tracking_state = False
                continue

            ALL_IDs = []  # create an empty array to be filled in the following
            # ALL_CENTROIDS = []  # create an empty array to be filled in the following
            ALL_bounding_boxes = []  # create an empty array to be filled in the following
            frame_number += 1  # counter from frame number

            # Draw standard Text
            # display and output
            text_positionUL = (pos_row, pos_col)  # cols, rows
            draw_standard_text(frame, frame_number, trackerType, text_positionUL)

            # draw tracked objects
            for m, newbox in enumerate(boxes):
                x, y, w, h = newbox[0], newbox[1], newbox[2], newbox[3]

                p7 = (int(x), int(y))
                p8 = (int(x + w), int(y + h))

                ID_counter = m + 1

                # Draw bbox
                cv2.rectangle(frame, p7, p8, colours[m], 2, 1)
                # Draw bbox's text
                draw_bbox_text(frame, colours[m], str(ID_counter) + ' - ' + classes[m], p7)

                # Coordinates of one box
                box_total = (p7, p8)

                ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
                # ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
                ALL_IDs.append(ID_counter)

            # Show frame
            print(f'Tracking step took "{time.time() - t1:.3f}" seconds.')

            # video_writer.write(frame)

            """ 
            https://stackoverflow.com/questions/51143458/difference-in-output-with-waitkey0-and-waitkey1/51143586

            1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
            2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
            So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1)
            the function will show a frame for 1 ms only.
            """
            waitKey = cv2.waitKey(delay_value)
            if waitKey & 0xFF == 27:
                end_task = True
                detection_state = False
                tracking_state = False
                print('\nButton Pressed: "Esc".\n')
                continue
            elif waitKey & 0xFF == ord('q'):
                end_task = True
                detection_state = False
                tracking_state = False
                print('\nButton Pressed: "q".\n')
                continue
            elif waitKey & 0xFF == ord('d'):  # 'd' pressed
                detection_state = True
                tracking_state = False
                print('\nButton Pressed: "d".\n')
                continue
            elif 0xFF == ord('D'):  # 'D' pressed
                detection_state = True
                tracking_state = False
                print('\nButton Pressed: "D".\n')
                continue
            else:
                cv2.imshow('YOLOv5 x ' + trackerType, frame)

                vid_writer.write(frame)

                continue
        else:
            print('Ending Detection and Tracking')
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()
            break


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
    parser.add_argument('--trackerType', type=str, default='CSRT',
                        help='Use one of the following: ["CSRT","MIL","KCF","TLD","MedianFlow", "GOTURN", "MOSSE","Boosting"]')
    opt_ = parser.parse_args()
    print(opt_)

    with torch.no_grad():
        if opt_.update:  # update all models (to fix SourceChangeWarning)
            for opt_.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect_and_track3(opt_)
                strip_optimizer(opt_.weights)
        else:
            detect_and_track3(opt_)
