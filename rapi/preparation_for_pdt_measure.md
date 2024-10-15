### Steps For measuring the "Object Selected by User" 

#### 1) Need to calibrate the camera for Sleeper Bed Environment
    
    1. Frame size to be fixed, after calibration
    2. Need to find the Lens Position (Focal Length):(for Pi)camera module 3 
        - To other camera, this setting might be different 
    3. Used specification as below:
        width = 1920
        height = 1080
        
        # Create a still configuration with the desired size
        still_config = picam2.create_still_configuration(main={"size": (width, height)})
        picam2.configure(still_config)
        picam2.set_controls({
            "LensPosition": 1.70,
            "AeEnable": True,
            "ExposureTime": 8500,
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
            "AnalogueGain": 1.0
        })
    4. Need to for sure calibrated using the chess board images to capture all the 
    edges, and under all position of the chess board inside the frame
    All this is done using chess_calibration.py

#### 2) Do the visual inspection of the calibration done in step 1:

    1. In order to check if the calibration is sufficient,
    and provides enough accuracy of edge detection, 
    2. we need to use visu_verify_undistora.py

#### 3) Need to find Pixel to MM size:
    1. Need to tell the script, what is the real world chess box dimension,
    2. Will find the ratio of Pixel to MM size
    3. Script will find the chess edge dimension in pixels, in a single square
    
    Note: The angle of the camera must be 90 deg for getting best accuracy 
    else, there is pixel measurement difference

    Note: Need to discuss on the video "Pixel_Chess_box_explanation.mp4"
    Script used is: Final_chess_verify.py

#### 4) Next step is to use the real world object, detect and do measurement. 
    1. use script like round_object.py to detect and get the measurement.  
    
### Next Step:

#### Objective: Write the script for detecting and measuring the object that is chosen by the user

#### Objects to use:
    1. Use to capture of the TV Remote full dimension, along with dimension of the buttons.


