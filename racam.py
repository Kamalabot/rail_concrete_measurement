import cv2
import numpy as np

# Load the saved calibration data
calibration_data = np.load('camera_calibration.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# Known square size on the chessboard (e.g., 25 mm)
square_size = 25.0  # mm

# Set the video stream URL for the Raspberry Pi camera
video_stream_url = 'http://192.168.1.42:5000/video_feed'  # Replace with your actual stream URL

# Start capturing images from the video stream
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get image dimensions
    h, w = frame.shape[:2]

    # Undistort the frame using the calibration data
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

    # Crop the undistorted frame to remove black borders, if necessary
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Convert the frame to grayscale for corner detection
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the image
    chessboard_size = (9, 6)  # (number of inner corners per chessboard row and column)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Draw the chessboard corners on the undistorted frame
        cv2.drawChessboardCorners(undistorted_frame, chessboard_size, corners_refined, ret)

        # Measure distance between the first two corners along a row
        pt1 = corners_refined[0][0]  # First corner in pixels
        pt2 = corners_refined[1][0]  # Second corner in pixels

        # Calculate the pixel distance between the two points
        pixel_distance = np.linalg.norm(pt1 - pt2)

        # Convert pixel distance to real-world distance (mm) using the known square size
        real_world_distance_mm = (pixel_distance / square_size) * square_size

        # Calculate the difference between the expected and actual distance
        expected_distance_mm = square_size  # Expected distance is 25mm (the size of one square)
        difference_mm = abs(real_world_distance_mm - expected_distance_mm)

        # Display the measured distance and difference on the frame
        cv2.putText(undistorted_frame, f"Measured: {real_world_distance_mm:.2f} mm", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(undistorted_frame, f"Difference: {difference_mm:.2f} mm", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame with measurements
    cv2.imshow('Measurement', undistorted_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
