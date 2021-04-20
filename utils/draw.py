import cv2
from tracking.tracking_constants import B_text, G_text, R_text
from datetime import datetime  # get current DateTime
from random import randint


def draw_bbox_text(frame, color, ID_counter, p7):
    cv2.putText(frame, 'Tracked_obj ID:', (p7[0] - 25, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color,
                1, cv2.LINE_AA)
    cv2.putText(frame, str(ID_counter), (p7[0] + 220, p7[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color,
                1, cv2.LINE_AA)


def draw_standard_text(frame, current_frame_number, trackerType, text_positionUL):
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


def draw_lost_trackings_text(frame, w, h):
    cv2.putText(frame, 'TRACKED OBJECT LOST', ((int(w / 4), int(h / 3))),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, 'RESTART THE TRACKER', (int(w / 4), int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)


# Only for the first frame
def draw_bboxes_with_mouse(frame):
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
    return bboxes, colors

