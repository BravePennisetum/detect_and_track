from abc import ABC
from abc import abstractmethod


class Detector(ABC):
    class_names: list

    def __init__(self, class_names: list):
        self.class_names = class_names

    # This method when implemented must return 2 lists
    # 1) bboxes list --> bbox = (x, y, w, h) with (x,y) being the top left corner of the bbox
    # 2) classes list
    # 3) confidence list
    # 4) colors for the bboxes
    @abstractmethod
    def detect(self, frame) -> (list, list, list, list):
        pass
