def initial_prints(starting_time, fps):
    print(
        "********************************  START OF VIDEO PROCESSING ************************************************************************************************")
    print("Multiple object tracking")
    print('Start time:', starting_time)

    ########################################################################################################
    # LOCATION FOR PARAMETERS AND PATHS FROM "dt_tracking_setting.py"
    ########################################################################################################

    print("FPS of the used camera = ", fps)
    print(
        "Draw a bounding box over an object and then press the enter key and then press any other key (except q key) to select next object")
    print("Press q to quit selecting boxes and start tracking")
