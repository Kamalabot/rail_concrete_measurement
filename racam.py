import cv2
import numpy as np

# Load calibration parameters from the .npz file
calibration_data = np.load('camera_calibration.npz')
mtx = calibration_data['mtx']
dist = calibration_data['dist']

# Chessboard dimensions
chessboard_size = (9, 6)  # Number of inner corners (columns, rows)
square_size_mm = 25       # Square size in millimeters

# Camera focal length (in mm)
focal_length_mm = 100  # Adjust based on your camera lens

# Distance from camera to chessboard (in mm)
working_distance_mm = 84  # Adjust based on your setup

# Column to measure (0-indexed, e.g., 0 for first column)
column_to_measure = 0

# Replace with your Raspberry Pi's stream address
raspberry_pi_ip = "http://192.168.1.42:5000/video_feed"

# Capture video from the Raspberry Pi stream
cap = cv2.VideoCapture(raspberry_pi_ip)

while True:
    # Read a frame from the stream
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for corner detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_frame, chessboard_size, None)

    if ret:
        # Refine the corners
        corners_refined = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # Draw the chessboard corners on the frame
        cv2.drawChessboardCorners(frame, chessboard_size, corners_refined, ret)

        # Measure distances for the selected column
        for i in range(chessboard_size[1] - 1):
            corner1 = corners_refined[i * chessboard_size[0] + column_to_measure]
            corner2 = corners_refined[(i + 1) * chessboard_size[0] + column_to_measure]
            pixel_distance = np.linalg.norm(corner1 - corner2)

            # Calculate the real-world distance using focal length and working distance
            estimated_mm = (pixel_distance * working_distance_mm) / focal_length_mm

            # Draw the measurement on the preview with better readability
            midpoint = (corner1 + corner2) / 2
            midpoint = tuple(midpoint.ravel().astype(int))

            # Draw text with black outline and white inside for readability
            text = f"{estimated_mm:.2f} mm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            # Draw outline (black)
            cv2.putText(frame, text, midpoint, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Draw text (white)
            cv2.putText(frame, text, midpoint, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Display the raw frame with measurements
    cv2.imshow("Raspberry Pi Stream with Measurements", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
