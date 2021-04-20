import cv2
import time

# videoPath = "videos/sample.avi"
iter_second_indicator = 1  # time shift in seconds to provide speed information ###############

pos_row = 10
pos_col = 50
pos_row_time = 10
pos_col_time = 20
B_text = 255
G_text = 255
R_text = 255
B_text_time = 255
G_text_time = 255
R_text_time = 255
font = cv2.FONT_HERSHEY_SIMPLE = 1  # font for displaced datetime
# fps = cam.get(cv2.CAP_PROP_FPS)  # OR SET IT MANUALLY
delay_value = 1  # for the waitkey

trackerTypes = ['CSRT', 'Boosting', 'MIL', 'KCF', 'TLD', 'MedianFlow', 'GOTURN', 'MOSSE']
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
