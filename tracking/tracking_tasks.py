from random import randint
from datetime import datetime  # get current DateTime
import numpy as np
from tracking.multi_tracker.tracker import createTrackerByName
import sys

from tracking.tracking_constants import *
from utils.general import write_to_txt, write_list_to_txt, Dirs
from utils.draw import draw_bbox_text, draw_standard_text


# Used for drawing bboxes with a mouse
def track(
        trackerType,
        videoPath,
        video_format,
        dataset_mode,
        dirs,
        starting_time,
        i_should_crop_images,
        crop_dir
):
    with open(dirs.get_command_dir() + '/event_information_' + starting_time + '.txt', 'w') as event_info_stream:
        # Create a video capture object to read videos
        cap = cv2.VideoCapture(videoPath)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if dataset_mode == 'video':
            if video_format == 'avi' or video_format == 'AVI':
                print('Choosing XVID codec')
                fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            elif video_format == 'mp4' or video_format == 'MP4':
                print('Choosing MP4V codec')
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            # Create video writer
            vid_writer = cv2.VideoWriter(
                dirs.get_out_path(),
                fourcc,
                fps,
                (frame_width, frame_height)
            )

        # Read first frame
        success, frame = cap.read()
        # quit if unable to read the video file
        if not success:
            write_to_txt(event_info_stream, 'Failed to read video')
            sys.exit(1)

        ## Select boxes and define colos
        bboxes = []
        colors = []

        # loop for selecting several objects
        # TODO - Modify to show button when done
        while True:
            # draw bounding boxes over objects
            bbox = cv2.selectROI('MultiTracker', frame)
            print(type(bbox), bbox)
            bboxes.append(bbox)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break

        # print('Selected bounding boxes {}'.format(bboxes))

        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()

        # Initialize MultiTracker - You can specify different trackers for every bounding box
        # TODO - Modify to support different trackers
        for bbox in bboxes:
            multiTracker.add(createTrackerByName(trackerType), frame, bbox)

        # Process video and track objects

        current_frame_number = -1  # initialize counter from frame number
        second_indicator_A = 0  # initialize the seconds to provide speed information###############
        second_indicator_B = iter_second_indicator  # initialize the seconds to provide speed information###############

        """ ------------- Start the main Loop ------------- """
        while cap.isOpened():
            # Read frame
            success, frame = cap.read()

            """ Uncomment if you want to crop images """
            if i_should_crop_images:
                success_bbb, frame_for_crop = cap.read()  # activate only if you want to crop images

            if not success:
                break
            # out_initial.write(frame) ##
            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame)

            if not success:
                write_to_txt(event_info_stream, "@@@@@@@@@@@@@@@@ TRACKED OBJECT LOST @@@@@@@@@@@@@@@@@@@@@")
                cv2.putText(frame, 'TRACKED OBJECT LOST', ((int(frame_width / 4), int(frame_height / 3))),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.putText(frame, 'RESTART THE TRACKER', (int(frame_width / 4), int(frame_height / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

            ALL_IDs = []  # create an empty array to be filled in the following
            ALL_CENTROIDS = []  # create an empty array to be filled in the following
            ALL_bounding_boxes = []  # create an empty array to be filled in the following
            current_frame_number = current_frame_number + 1  # counter from frame number

            # draw tracked objects
            for m, newbox in enumerate(boxes):
                p7 = (int(newbox[0]), int(newbox[1]))
                p8 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p7, p8, colors[m], 2, 1)

                # display and output
                text_positionUL = (pos_row, pos_col)  # cols, rows
                # Coordinates of one box
                box_total = (p7, p8)
                box_centroid_x = int((p8[0] - p7[0]) / 2) + p7[0]
                box_centroid_y = int((p8[1] - p7[1]) / 2) + p7[1]
                box_centr = (box_centroid_x, box_centroid_y)  # centroid of the box where x (as cols),y (as rows)

                ID_counter = m + 1

                ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
                ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
                ALL_IDs.append(ID_counter)

                # Crop image for each object
                if i_should_crop_images:
                    crop_img = frame_for_crop[p7[1]:p8[1], p7[0]:p8[0]]
                    pixel_area = (p8[0] - p7[0]) * (p8[1] - p7[1])
                    pixel_max = int(crop_img.max())
                    pixel_min = int(crop_img.min())
                    pixel_mean = int(crop_img.mean())
                    pixel_std = int(crop_img.std())
                    cr_metadata = np.array((pixel_area, pixel_max, pixel_min, pixel_mean, pixel_std))

                    cv2.imwrite(crop_dir + '/frame_' + str(current_frame_number) + '_ID' + str(ID_counter) + '.jpg',
                                crop_img)
                    np.savetxt(
                        fname=crop_dir + '/frame_' + str(current_frame_number) + '_ID' + str(ID_counter) + '.txt',
                        X=cr_metadata,
                        fmt='%.f',
                        delimiter=',',
                        header="Area,Max,Min,Mean,Std"
                    )

                # Draw text(s) on frame
                cv2.putText(frame, str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
                            (text_positionUL[0], text_positionUL[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (B_text, G_text, R_text), 1, cv2.LINE_AA)
                cv2.putText(frame, 'frame:', (text_positionUL[0], text_positionUL[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (B_text, G_text, R_text), 1, cv2.LINE_AA)
                cv2.putText(frame, str(current_frame_number), (text_positionUL[0] + 50, text_positionUL[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B_text, G_text, R_text), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Tracker: ' + trackerType, (text_positionUL[0], text_positionUL[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (B_text, G_text, R_text), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Tracked_obj ID:', (p7[0] - 25, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colors[m],
                            1, cv2.LINE_AA)
                cv2.putText(frame, str(ID_counter), (p7[0] + 220, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colors[m],
                            1, cv2.LINE_AA)

                # Show frame
                cv2.imshow('MultiTracker', frame)

            # out.write(frame)

            # Write to txt file
            # TODO - also write to csv with keys: a) video_name --> str, b) frame --> int, c) object_IDs --> list of ints, d) bboxes --> list of ints, e) centroids --> list of ints
            write_list_to_txt(
                event_info_stream,
                [
                    'Processing frame: ' + str(current_frame_number),  # current frame number
                    str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),  # datetime
                    'Object IDs: ' + str(ALL_IDs),
                    'Bounding boxes pixels: ' + str(ALL_bounding_boxes),
                    # Bounding box --> UP left x (as cols),y (as rows) and BR x (as cols),y (as rows)
                    'Centroids pixels: ' + str(ALL_CENTROIDS),
                    '--------------------------------------------------'
                ]
            )

            if dataset_mode == 'image':
                # Save frame as image
                try:
                    cv2.imwrite(dirs.get_extracted_video_dir() + '/f (' + str(current_frame_number) + ')' + '.jpg',
                                frame)  # *****************.
                except Exception as e:
                    write_to_txt(event_info_stream, 'Failed to save "frame" --> ' + str(e))

                if i_should_crop_images:
                    try:
                        cv2.imwrite(dirs.get_streaming_frames_dir() + '/f (' + str(current_frame_number) + ')' + '.jpg',
                                    frame_for_crop)  # *****************
                    except Exception as e:
                        write_to_txt(event_info_stream, 'Failed to save "frame_for_crop" --> ' + str(e))
            else:  # 'video'
                vid_writer.write(frame)

            # quit on ESC button
            if cv2.waitKey(delay_value) & 0xFF == 27:  # Esc pressed
                break
                # out.release()
                # out_initial.release()
        cap.release()
        if dataset_mode == 'video':
            vid_writer.release()
            print('Released video writer.')
        cv2.destroyAllWindows()
        write_to_txt(event_info_stream,
                     "********************************  END OF VIDEO PROCESSING ********************************")


# Used for getting bboxes from outside this function - It can be used with a detector or a drawing bboxes mechanismq
def track2(
        cvCapture: cv2.VideoCapture,
        frame,
        current_frame_number: int,
        bboxes: list,
        colors: list,
        trackerType: str,
        dirs: Dirs,
        video_writer: cv2.VideoWriter
):
    """
    :param cvCapture: A 'cv2.Capture' object --> used to process source of video of interest
    :param frame: current frame. This should be the frame after detection
    :param current_frame_number: current frame number.
    :param bboxes: list of bounding boxes
    :param trackerType: default is 'CSRT'. Other options: 'Boosting', 'MIL', 'KCF', 'TLD', 'MedianFlow', 'GOTURN', 'MOSSE'
    :param dirs: A Dirs object that creates directories to save the results of the processes
    :param video_writer: A 'cv2.VideoWriter' object --> should be passed initialized. Used to write frames.
    :return:
    """
    end_task = False
    # Log everything
    with open(dirs.get_command_dir() + '/tracking_information_frame-' + str(current_frame_number) + '.txt', 'w') as event_info_stream:
        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()

        # Initialize MultiTracker - You can specify different trackers for every bounding box
        # TODO - Modify to support different trackers
        for bbox, color in zip(bboxes, colors):
            # rect = (x, y, w, h)
            rect = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            # rect = cv2.rectangle(frame, pt1, pt2, color)
            print(type(rect), rect)
            multiTracker.add(createTrackerByName(trackerType), frame, rect)

            frame_width = int(cvCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cvCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cvCapture.get(cv2.CAP_PROP_FPS)
            print('(Frame width, Frame height, FPS) = ({w}, {h}, {fps})'.format(w=frame_width, h=frame_height, fps=fps))

            """ ------------- Start the main Loop ------------- """
        while cvCapture.isOpened():

            # Read frame
            success, frame = cvCapture.read()
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
                    "@@@@@@@@@@@@@@@@ TRACKED OBJECT LOST @@@@@@@@@@@@@@@@@@@@@ at frame:" + str(current_frame_number)
                )
                # TODO - putText for lost objects outside of the frame of the video - otherwise there won't be a way to remove the "putText"
                # draw_lost_trackings_text(frame, frame_width, frame_height)
                end_task = False
                break

            ALL_IDs = []  # create an empty array to be filled in the following
            # ALL_CENTROIDS = []  # create an empty array to be filled in the following
            ALL_bounding_boxes = []  # create an empty array to be filled in the following
            current_frame_number += 1  # counter from frame number

            # Draw standard Text
            # display and output
            text_positionUL = (pos_row, pos_col)  # cols, rows
            draw_standard_text(frame, current_frame_number, trackerType, text_positionUL)

            # draw tracked objects
            for m, newbox in enumerate(boxes):
                x, y, w, h = newbox[0], newbox[1], newbox[2], newbox[3]

                p7 = (int(x), int(y))
                p8 = (int(x + w), int(y + h))

                cv2.rectangle(frame, p7, p8, colors[m], 2, 1)

                # Coordinates of one box
                box_total = (p7, p8)

                ID_counter = m + 1

                ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
                # ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
                ALL_IDs.append(ID_counter)

                # Draw bbox_text
                draw_bbox_text(frame, colors[m], ID_counter, p7)

            # Show frame
            cv2.imshow('Detector x MultiTracker', frame)
            print(f'Tracking step took "{time.time() -t1:.3f}" seconds.')

            # Write to txt file
            # TODO - also write to csv with keys: a) video_name --> str, b) frame --> int, c) object_IDs --> list of ints, d) bboxes --> list of ints, e) centroids --> list of ints
            write_list_to_txt(
                event_info_stream,
                [
                    'Processing frame: ' + str(current_frame_number),  # current frame number
                    str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),  # datetime
                    'Object IDs: ' + str(ALL_IDs),
                    'Bounding boxes pixels: ' + str(ALL_bounding_boxes),
                    # Bounding box --> UP left x (as cols),y (as rows) and BR x (as cols),y (as rows)
                    # 'Centroids pixels: ' + str(ALL_CENTROIDS),
                    '--------------------------------------------------'
                ]
            )

            # Write frame to output video
            video_writer.write(frame)

            # quit on ESC button
            if cv2.waitKey(delay_value) & 0xFF == 27:  # Esc pressed
                end_task = True
                break

            """ 
            https://stackoverflow.com/questions/51143458/difference-in-output-with-waitkey0-and-waitkey1/51143586
            
            1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
            2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
            So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1)
            the function will show a frame for 1 ms only.
            """
            if cv2.waitKey(delay_value) & 0xFF == ord('d'):  # 'd' pressed
                end_task = False
                break
            if cv2.waitKey(delay_value) & 0xFF == ord('D'):  # 'D' pressed
                end_task = False
                break

    return current_frame_number, end_task


# Original Structure
def track3(
        trackerType,
        videoPath,
        dirs,
        starting_time,
        i_should_crop_images,
        crop_dir
):
    # Set Standard Output to event_information_$starting_time$.txt
    sys.stdout = open(dirs.dir_command + '/event_information_' + starting_time + '.txt', 'w')

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # out = cv2.VideoWriter(outputvideoPath, fourcc, float(fps), (frame_width,frame_height))
    # out_initial = cv2.VideoWriter(outputvideoPath_streaming, fourcc, float(fps), (frame_width,frame_height))

    # Read first frame
    success, frame = cap.read()
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    ## Select boxes and define colos
    bboxes = []
    colors = []

    # loop for selecting several objects
    # TODO - Modify to show button when done
    while True:
        # draw bounding boxes over objects
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

    # print('Selected bounding boxes {}'.format(bboxes))

    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker - You can specify different trackers for every bounding box
    # TODO - Modify to support different trackers
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)

    # Process video and track objects

    current_frame_number = -1  # initialize counter from frame number
    second_indicator_A = 0  # initialize the seconds to provide speed information###############
    second_indicator_B = iter_second_indicator  # initialize the seconds to provide speed information###############

    """ ------------- Start the main Loop ------------- """
    while cap.isOpened():
        # Read frame
        success, frame = cap.read()

        """ Uncomment if you want to crop images """
        if i_should_crop_images:
            success_bbb, frame_for_crop = cap.read()  # activate only if you want to crop images

        if not success:
            break
        # out_initial.write(frame) ##
        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        if not success:
            print("@@@@@@@@@@@@@@@@ TRACKED OBJECT LOST @@@@@@@@@@@@@@@@@@@@@")
            cv2.putText(frame, 'TRACKED OBJECT LOST', ((int(frame_width / 4), int(frame_height / 3))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, 'RESTART THE TRACKER', (int(frame_width / 4), int(frame_height / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

        ALL_IDs = []  # create an empty array to be filled in the following
        ALL_CENTROIDS = []  # create an empty array to be filled in the following
        ALL_bounding_boxes = []  # create an empty array to be filled in the following
        current_frame_number = current_frame_number + 1  # counter from frame number

        # draw tracked objects
        for m, newbox in enumerate(boxes):
            p7 = (int(newbox[0]), int(newbox[1]))
            p8 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p7, p8, colors[m], 2, 1)

            # display and output
            text_positionUL = (pos_row, pos_col)  # cols, rows
            # Coordinates of one box
            box_total = (p7, p8)
            box_centroid_x = int((p8[0] - p7[0]) / 2) + p7[0]
            box_centroid_y = int((p8[1] - p7[1]) / 2) + p7[1]
            box_centr = (box_centroid_x, box_centroid_y)  # centroid of the box where x (as cols),y (as rows)

            ID_counter = m + 1

            ALL_bounding_boxes.append(box_total)  # fill the array during the iteration
            ALL_CENTROIDS.append(box_centr)  # fill the array during the iteration
            ALL_IDs.append(ID_counter)

            # Crop image for each object
            if i_should_crop_images:
                crop_img = frame_for_crop[p7[1]:p8[1], p7[0]:p8[0]]
                pixel_area = (p8[0] - p7[0]) * (p8[1] - p7[1])
                pixel_max = int(crop_img.max())
                pixel_min = int(crop_img.min())
                pixel_mean = int(crop_img.mean())
                pixel_std = int(crop_img.std())
                cr_metadata = np.array((pixel_area, pixel_max, pixel_min, pixel_mean, pixel_std))

                cv2.imwrite(crop_dir + '/frame_' + str(current_frame_number) + '_ID' + str(ID_counter) + '.jpg',
                            crop_img)
                np.savetxt(
                    fname=crop_dir + '/frame_' + str(current_frame_number) + '_ID' + str(ID_counter) + '.txt',
                    X=cr_metadata,
                    fmt='%.f',
                    delimiter=',',
                    header="Area,Max,Min,Mean,Std"
                )

            # Draw text(s) on frame
            cv2.putText(frame, str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")),
                        (text_positionUL[0], text_positionUL[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (B_text, G_text, R_text), 1, cv2.LINE_AA)
            cv2.putText(frame, 'frame:', (text_positionUL[0], text_positionUL[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (B_text, G_text, R_text), 1, cv2.LINE_AA)
            cv2.putText(frame, str(current_frame_number), (text_positionUL[0] + 50, text_positionUL[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B_text, G_text, R_text), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Tracker: ' + trackerType, (text_positionUL[0], text_positionUL[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (B_text, G_text, R_text), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Tracked_obj ID:', (p7[0] - 25, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colors[m],
                        1, cv2.LINE_AA)
            cv2.putText(frame, str(ID_counter), (p7[0] + 220, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colors[m],
                        1, cv2.LINE_AA)

            # Show frame
            cv2.imshow('MultiTracker', frame)

        # out.write(frame)

        # Save frame as image
        try:
            cv2.imwrite(dirs.dir_extracted_video + '/f (' + str(current_frame_number) + ')' + '.jpg',
                        frame)  # *****************.
        except Exception as e:
            print('Failed to save "frame" -->', e)

        if i_should_crop_images:
            try:
                cv2.imwrite(dirs.dir_streaming_frames + '/f (' + str(current_frame_number) + ')' + '.jpg',
                            frame_for_crop)  # *****************
            except Exception as e:
                print('Failed to save "frame_for_crop" -->', e)

        print('Processing frame: ', current_frame_number)  # current frame number
        print(str(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))  # datetime
        print('Object IDs: ', ALL_IDs)
        print('Bounding boxes pixels: ',
              ALL_bounding_boxes)  # Bounding box --> UP left x (as cols),y (as rows) and BR x (as cols),y (as rows)
        print('Centroids pixels: ', ALL_CENTROIDS)
        print('--------------------------------------------------')

        # quit on ESC button
        if cv2.waitKey(delay_value) & 0xFF == 27:  # Esc pressed
            # out.release()
            # out_initial.release()
            cap.release()
            cv2.destroyAllWindows()
            print(
                "********************************  END OF VIDEO PROCESSING ************************************************************************************************")
            break
    sys.stdout.close()
