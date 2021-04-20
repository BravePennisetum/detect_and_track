from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from detection.yolov5.models.experimental import attempt_load
from detection.yolov5.utils.datasets import convert_image
from detection.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh2, set_logging, \
    increment_path
from detection.yolov5.utils.torch_utils import select_device, time_synchronized
from datetime import datetime  # get current DateTime
from tracking.multi_tracker.tracker import createTrackerByName

from tracking.tracking_constants import *
from utils.general import write_to_txt, write_list_to_txt, expand_bboxes
from utils.draw import draw_bbox_text, draw_standard_text


class ArgParser:
    def __init__(
            self,
            weights='seagull_trained.pt',
            source='videos/sample.avi',
            img_size=640,
            conf_thres=0.25,
            iou_thres=0.45,
            device='0',  # 0 is for cuda
            view_img=False,
            save_txt=True,
            save_conf=True,
            classes=None,
            agnostic_nms=True,
            augment=True,
            update=False,
            project='detected/',
            name='detected/',
            exist_ok=True
    ):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        if classes is None:
            self.classes = [0]
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok


class YOLOv5_x_CSRT:

    # TODO - Fix the self.opt and the object variables
    def __init__(
            self,
            weights='seagull_trained.pt',
            source='videos/sample.avi',
            img_size=640,
            trackerType='CSRT',
            cam=None,
            video_writer=None,
            device='cpu',  # 0 is for cuda
            dirs=None,
            expansion_constant=20,
            conf_thres=0.25,
            iou_thres=0.45,
            view_img=False,
            save_txt=True,
            save_conf=True,
            classes=None,
            agnostic_nms=True,
            augment=True,
            update=False,
            project='detected/',
            name='detected/',
            save_dir='tracked/sample.avi - current date',
            exist_ok=True
    ):
        self.detection_state = True
        self.initiate_tracking = False
        self.cam = cam
        self.video_writer = video_writer
        self.trackerType = trackerType
        self.dirs = dirs
        self.expansion_constant = expansion_constant
        self.opt = ArgParser(
            weights,
            source,
            img_size,
            conf_thres, iou_thres, device, view_img, save_txt, save_conf, classes, agnostic_nms, augment, update,
            project,
            name, exist_ok
        )
        self.save_dir = save_dir
        self.__initializations(self.opt)

    def __initializations(self, opt):
        self.source, self.weights, self.view_img, self.save_txt, self.imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model

        # Set Image Re-size
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
        else:
            self.save_img = True

    def detect(self, frame, frame_number, save_img=True):

        print(f'---------------- Started Detection of Frame-{frame_number} ----------------')

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

        # Reshape input frame
        img, im0 = convert_image(frame, self.imgsz)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt.augment)[0]
        print(f'Model run in "{time.time() - t1:.3f}" seconds.')

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        t2 = time_synchronized()
        print(f'NMS ended in "{time.time() - t2:.3f}" seconds.')
        bboxes = []
        colours = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for xyxy in det:
                    colours.append(
                        (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
                    )
                    bboxes.append(xyxy2xywh2(xyxy))

        if self.save_txt or save_img:
            s = f"\n{len(list(Path(self.save_dir).glob('labels/*.txt')))} labels saved to {self.save_dir}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}{s}")

        print(f'Detection pipeline ended in "{time.time() - t0:.3f}" seconds.\n----------------------')
        return bboxes, colours

    # TODO - fix logic mistake
    def detect_and_track(self):
        frame_number = 0
        bboxes = []
        colours = []

        with open(self.dirs.get_command_dir() + '/tracking_information_frame-' + str(frame_number) + '.txt',
                  'w') as event_info_stream:
            if self.detection_state:
                print('Detection...............')
                while self.cam.isOpened():
                    success, frame = self.cam.read()
                    if not success:
                        print('End of video. Exiting...')
                        return

                    frame_number += 1
                    bboxes = []
                    colours = []
                    print(f'---------------- Started Detection in Frame-{frame_number} ----------------')
                    # Run inference
                    t0 = time.time()
                    img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
                    _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

                    # Reshape input frame
                    img, im0 = convert_image(frame, self.imgsz)

                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = self.model(img, augment=self.opt.augment)[0]
                    print(f'Model run in "{time.time() - t1:.3f}" seconds.')

                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)
                    t2 = time_synchronized()
                    print(f'NMS ended in "{time.time() - t2:.3f}" seconds.')

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            for xyxy in det:
                                colours.append(
                                    (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
                                )
                                bboxes.append(xyxy2xywh2(xyxy))

                    print(f'Detection pipeline ended in "{time.time() - t0:.3f}" seconds.\n----------------------')

                    if cv2.waitKey(delay_value) & 0xFF == ord('t'):  # 'd' pressed
                        self.detection_state = False
                        break

                    if cv2.waitKey(delay_value) & 0xFF == ord('T'):  # 'd' pressed
                        self.detection_state = False
                        break
                    cv2.imshow('Detector', frame)
            else:
                success, frame = self.cam.read()
                if not success:
                    print('End of video. Exiting...')
                    return

                if len(bboxes) <= 0:
                    print('No objects detected. Entering Detection mode...')
                    self.detection_state = True
                """ Tracking Part"""
                print(f'---------------- Started Tracking in Frame-{frame_number} ----------------')
                # Log everything

                # Create MultiTracker object
                multiTracker = cv2.MultiTracker_create()

                frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # fps = self.cam.get(cv2.CAP_PROP_FPS)
                # print('(Frame width, Frame height, FPS) = ({w}, {h}, {fps})'.format(w=frame_width, h=frame_height, fps=fps))
                bboxes = expand_bboxes(bboxes, frame_width, frame_height, c=self.expansion_constant)
                # Initialize MultiTracker - You can specify different trackers for every bounding box
                for bbox, color in zip(bboxes, colours):
                    rect = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    multiTracker.add(createTrackerByName(self.trackerType), frame, rect)

                """ ------------- Start the main Loop ------------- """
                while self.cam.isOpened():

                    # Read frame
                    success, frame = self.cam.read()
                    frame_number += 1
                    if not success:
                        write_to_txt(event_info_stream, "Video reached the end...")
                        break

                    # get updated location of objects in subsequent frames
                    t1 = time.time()
                    success, boxes = multiTracker.update(frame)
                    # if object is lost go to re-detect
                    if not success:
                        write_to_txt(
                            event_info_stream,
                            "@@@@@@@@@@@@@@@@ TRACKED OBJECT LOST @@@@@@@@@@@@@@@@@@@@@ at frame:" + str(frame_number)
                        )
                        # TODO - putText for lost objects outside of the frame of the video - otherwise there won't be a way to remove the "putText"
                        # draw_lost_trackings_text(frame, frame_width, frame_height)
                        end_task = False
                        break

                    ALL_IDs = []  # create an empty array to be filled in the following
                    # ALL_CENTROIDS = []  # create an empty array to be filled in the following
                    ALL_bounding_boxes = []  # create an empty array to be filled in the following
                    frame_number += 1  # counter from frame number

                    # Draw standard Text
                    # display and output
                    text_positionUL = (pos_row, pos_col)  # cols, rows
                    draw_standard_text(frame, frame_number, self.trackerType, text_positionUL)

                    # draw tracked objects
                    for m, newbox in enumerate(boxes):
                        x, y, w, h = newbox[0], newbox[1], newbox[2], newbox[3]

                        p7 = (int(x), int(y))
                        p8 = (int(x + w), int(y + h))

                        cv2.rectangle(frame, p7, p8, colours[m], 2, 1)

                        # Coordinates of one box
                        box_total = (p7, p8)

                        ID_counter = m + 1

                        ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
                        # ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
                        ALL_IDs.append(ID_counter)

                        # Draw bbox_text
                        draw_bbox_text(frame, colours[m], ID_counter, p7)

                    # Show frame
                    print(f'Tracking step took "{time.time() - t1:.3f}" seconds.')

                    # Write to txt file
                    # TODO - also write to csv with keys: a) video_name --> str, b) frame --> int, c) object_IDs --> list of ints, d) bboxes --> list of ints, e) centroids --> list of ints
                    write_list_to_txt(
                        event_info_stream,
                        [
                            'Processing frame: ' + str(frame_number),  # current frame number
                            str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),  # datetime
                            'Object IDs: ' + str(ALL_IDs),
                            'Bounding boxes pixels: ' + str(ALL_bounding_boxes),
                            # Bounding box --> UP left x (as cols),y (as rows) and BR x (as cols),y (as rows)
                            # 'Centroids pixels: ' + str(ALL_CENTROIDS),
                            '--------------------------------------------------'
                        ]
                    )

                    # Write frame to output video
                    self.video_writer.write(frame)

                    """ 
                    https://stackoverflow.com/questions/51143458/difference-in-output-with-waitkey0-and-waitkey1/51143586

                    1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
                    2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
                    So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1)
                    the function will show a frame for 1 ms only.
                    """
                    # quit on ESC button
                    if cv2.waitKey(delay_value):
                        if 0xFF == 27:  # Esc pressed
                            return
                        elif 0xFF == ord('d'):  # 'd' pressed
                            self.detection_state = True
                            break
                        elif 0xFF == ord('D'):  # 'D' pressed
                            self.detection_state = True
                            break
                        else:
                            continue
            cv2.imshow('YOLOv5 x ' + self.trackerType, frame)
        return

    def detect_and_track2(self):
        frame_number = 0
        bboxes = []
        colours = []
        end_task = False
        just_because = 0
        # START LOGGER
        with open(self.dirs.get_command_dir() + '/tracking_information_frame-' + str(frame_number) + '.txt',
                  'w') as event_info_stream:
            # LOOP while 'camera' is open
            while self.cam.isOpened():
                # When Escape is pressed
                if end_task:
                    break
                # Read 1 Frame
                success, frame = self.cam.read()
                frame_number += 1
                if not success:
                    write_to_txt(event_info_stream, "Video reached the end...")
                    print('End of video. Exiting...')
                    break

                """ ----------  DETECTION ----------  """
                if self.detection_state:
                    print('Detection...............')
                    bboxes = []
                    colours = []
                    print(f'---------------- Started Detection in Frame-{frame_number} ----------------')
                    # Run inference
                    t0 = time.time()
                    img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
                    _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

                    # Reshape input frame
                    img, im0 = convert_image(frame, self.imgsz)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = self.model(img, augment=self.opt.augment)[0]
                    print('Predictions:\n', pred)
                    print(f'Model run in "{time.time() - t1:.3f}" seconds.')

                    t2 = time.time()
                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)
                    time_synchronized()
                    print(f'NMS ended in "{time.time() - t2:.3f}" seconds.')

                    # Process predictions
                    t3 = time.time()
                    for i, det in enumerate(pred):  # detections per image

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            for xyxy in det:
                                colours.append(
                                    (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
                                )
                                bboxes.append(xyxy2xywh2(xyxy))
                            self.detection_state = False
                            self.initiate_tracking = True
                    print(f'Decoding predictions ended in "{time.time() - t3:.3f}" seconds.')

                    print(f'Detection pipeline ended in "{time.time() - t0:.3f}" seconds.\n----------------------')

                    waitKey = cv2.waitKey(delay_value)
                    """ It does not make sense now that I think about it, but there might be a fix """
                    # TODO - add as a feature to run only detection
                    if waitKey & 0xFF == ord('t'):  # 'd' pressed
                        self.detection_state = False
                        self.initiate_tracking = True
                        continue
                    if waitKey & 0xFF == ord('T'):  # 'd' pressed
                        self.detection_state = False
                        self.initiate_tracking = True
                        continue
                    # cv2.imshow('Detector', frame)
                else:
                    """ ----------  TRACKING ----------  """
                    success, frame = self.cam.read()

                    if len(bboxes) <= 0:
                        print('No objects detected. Entering Detection mode...')
                        self.detection_state = True
                        continue
                    """ Tracking Part"""
                    print(f'---------------- Started Tracking in Frame-{frame_number} ----------------')
                    # Log everything

                    frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # The trackers work better if the bounding box is bigger than the object itself
                    bboxes = expand_bboxes(bboxes, frame_width, frame_height, c=self.expansion_constant)

                    # Create MultiTracker object - recreate in order to re-enter new bboxes.
                    if self.initiate_tracking:
                        multiTracker = cv2.MultiTracker_create()

                        # Initialize MultiTracker - You can specify different trackers for every bounding box
                        for bbox, color in zip(bboxes, colours):
                            rect = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                            multiTracker.add(createTrackerByName(self.trackerType), frame, rect)

                        self.initiate_tracking = False

                    """ ------------- Start the main Loop ------------- """
                    # get updated location of objects in subsequent frames
                    t1 = time.time()
                    success, boxes = multiTracker.update(frame)
                    # if object is lost go to re-detect
                    if not success:
                        write_to_txt(
                            event_info_stream,
                            "@@@@@@@@@@@@@@@@ TRACKED OBJECT LOST @@@@@@@@@@@@@@@@@@@@@ at frame:" + str(frame_number)
                        )
                        # TODO - putText for lost objects outside of the frame of the video - otherwise there won't be a way to remove the "putText"
                        # draw_lost_trackings_text(frame, frame_width, frame_height)
                        self.detection_state = True
                        end_task = False
                        continue

                    ALL_IDs = []  # create an empty array to be filled in the following
                    # ALL_CENTROIDS = []  # create an empty array to be filled in the following
                    ALL_bounding_boxes = []  # create an empty array to be filled in the following
                    frame_number += 1  # counter from frame number

                    # Draw standard Text
                    # display and output
                    text_positionUL = (pos_row, pos_col)  # cols, rows
                    draw_standard_text(frame, frame_number, self.trackerType, text_positionUL)

                    # draw tracked objects
                    for m, newbox in enumerate(boxes):
                        x, y, w, h = newbox[0], newbox[1], newbox[2], newbox[3]

                        p7 = (int(x), int(y))
                        p8 = (int(x + w), int(y + h))

                        ID_counter = m + 1

                        # Draw bbox
                        cv2.rectangle(frame, p7, p8, colours[m], 2, 1)
                        # Draw bbox's text
                        draw_bbox_text(frame, colours[m], ID_counter, p7)

                        # Coordinates of one box
                        box_total = (p7, p8)

                        ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
                        # ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
                        ALL_IDs.append(ID_counter)

                    # Show frame
                    print(f'Tracking step took "{time.time() - t1:.3f}" seconds.')

                    # Write to txt file
                    # TODO - also write to csv with keys: a) video_name --> str, b) frame --> int, c) object_IDs --> list of ints, d) bboxes --> list of ints, e) centroids --> list of ints
                    write_list_to_txt(
                        event_info_stream,
                        [
                            'Processing frame: ' + str(frame_number),  # current frame number
                            str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),  # datetime
                            'Object IDs: ' + str(ALL_IDs),
                            'Bounding boxes pixels: ' + str(ALL_bounding_boxes),
                            # Bounding box --> UP left x (as cols),y (as rows) and BR x (as cols),y (as rows)
                            # 'Centroids pixels: ' + str(ALL_CENTROIDS),
                            '--------------------------------------------------'
                        ]
                    )

                    # Write frame to output video
                    self.video_writer.write(frame)

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
                    continue
                elif waitKey & 0xFF == ord('d'):  # 'd' pressed
                    self.detection_state = True
                    continue
                elif 0xFF == ord('D'):  # 'D' pressed
                    self.detection_state = True
                    continue
                else:
                    cv2.imshow('YOLOv5 x ' + self.trackerType, frame)
                    continue
        return

