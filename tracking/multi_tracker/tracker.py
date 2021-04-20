from cv2 import TrackerCSRT_create
from cv2 import TrackerMIL_create
from cv2 import TrackerKCF_create
from cv2 import TrackerTLD_create
from cv2 import TrackerMedianFlow_create
from cv2 import TrackerGOTURN_create
from cv2 import TrackerMOSSE_create
from cv2 import TrackerBoosting_create


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == 'CSRT':
        tracker = TrackerCSRT_create()
    elif trackerType == 'MIL':
        tracker = TrackerMIL_create()
    elif trackerType == 'KCF':
        tracker = TrackerKCF_create()
    elif trackerType == 'TLD':
        tracker = TrackerTLD_create()
    elif trackerType == 'MedianFlow':
        tracker = TrackerMedianFlow_create()
    elif trackerType == 'GOTURN':
        tracker = TrackerGOTURN_create()
    elif trackerType == 'MOSSE':
        tracker = TrackerMOSSE_create()
    elif trackerType == 'Boosting':
        tracker = TrackerBoosting_create()
    else:
        tracker = None
        print('Incorrect tracker name')

    return tracker
