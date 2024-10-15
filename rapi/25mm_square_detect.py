import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls
import threading
import time

# Load the saved calibration data from the npz file
calibration_data = np.load('camera_calibration_data.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Initialize camera
picam2 = Picamera2()
width = 1920
height = 1080

# Create a still configuration with the desired size
still_config = picam2.create_still_configuration(main={"size": (width, height)})
picam2.configure(still_config)

# Set manual controls for exposure and gain
picam2.set_controls({
    "LensPosition": 1.70,                          # Manual focus
    "AeEnable": True,                              # Auto exposure disabled for manual control
    "ExposureTime": 8500,                          # Set shutter speed to 8500 μs (8.5 ms)
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
    "AnalogueGain": 1.0                            # Set low gain for reduced noise
})

# Known pixel-to-mm ratio (from your calibration data)
pixel_to_mm_ratio = 0.228  # Example: 0.228 mm per pixel, adjust based on your calibration

# Chessboard pattern size (number of inner corners per row and column)
chessboard_size = (9, 6)  # 9x6 chessboard (8 squares across, 5 squares down)

# Start camera
picam2.start()

print("Press 'q' to quit.")

# A lock for multithreaded access
frame_lock = threading.Lock()

# Flag to indicate when to stop the detection thread
stop_thread = False

# Variable to store the latest processed frame for live display
processed_frame = None

# Function to handle chessboard detection
def detect_chessboard():
    global processed_frame, stop_thread
    while not stop_thread:
        # Capture frame in RGBA format
        with frame_lock:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Undistort the captured image using the calibration data
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Convert to grayscale for corner detection
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner positions for better accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Draw and display the chessboard corners
            cv2.drawChessboardCorners(undistorted_image, chessboard_size, corners, ret)

            # Measure and draw each square
            for row in range(chessboard_size[1] - 1):  # Loop through rows
                for col in range(chessboard_size[0] - 1):  # Loop through columns
                    # Get the four corners of the current square
                    top_left = corners[row * chessboard_size[0] + col][0]
                    top_right = corners[row * chessboard_size[0] + col + 1][0]
                    bottom_left = corners[(row + 1) * chessboard_size[0] + col][0]
                    bottom_right = corners[(row + 1) * chessboard_size[0] + col + 1][0]

                    # Calculate the width and height of the square in pixels
                    width_px = np.linalg.norm(top_left - top_right)
                    height_px = np.linalg.norm(top_left - bottom_left)

                    # Calculate the average size of the square (in pixels and mm)
                    avg_pixel_size = (width_px + height_px) / 2
                    real_world_size_mm = avg_pixel_size * pixel_to_mm_ratio

                    # Draw the square (use cv2.polylines or cv2.rectangle)
                    square_contour = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
                    cv2.polylines(undistorted_image, [square_contour], isClosed=True, color=(0, 0, 255), thickness=2)

                    # Calculate the center of the square
                    center_x = int((top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4)
                    center_y = int((top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4)

                    # Display the size in mm at the center of the square
                    cv2.putText(undistorted_image, f"{real_world_size_mm:.2f}mm", (center_x - 30, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            print(f"Detected {chessboard_size[0] - 1}x{chessboard_size[1] - 1} squares.")

        else:
            # If the chessboard is not detected, display a message
            cv2.putText(undistorted_image, "Chessboard not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        with frame_lock:
            processed_frame = undistorted_image

        time.sleep(0.1)  # Slight delay to reduce CPU usage

# Start the chessboard detection in a separate thread
detection_thread = threading.Thread(target=detect_chessboard)
detection_thread.start()

while True:
    # Display the latest processed frame
    with frame_lock:
        if processed_frame is not None:
            cv2.imshow('Chessboard Square Measurement', processed_frame)

    # Wait for key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        stop_thread = True  # Signal the detection thread to stop
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()

# Wait for the detection thread to finish
detection_thread.join()
