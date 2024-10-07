# Steps to Measure Real-World Size Using Camera Calibration:
# Calibrate the Camera:

# First, you need to calibrate the camera using a known pattern like a chessboard. This gives you the camera matrix and distortion coefficients, which are essential for accurate measurements.
# Undistort the Image:

# Use the calibration data to undistort the image, ensuring that lens distortions don't affect the measurements.
# Calculate Real-World Dimensions:

# Once you have an undistorted image, you can use the intrinsic parameters from the camera calibration to map pixel coordinates to real-world coordinates.

import cv2
import numpy as np


# Function to capture and process video frames
def capture_and_measure_square(
    camera_matrix, dist_coeffs, square_size_real_world, pattern_size
):
    # Open a connection to the camera (camera index 0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Get frame size
        h, w = frame.shape[:2]

        # Get optimal new camera matrix (after calibration)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort the frame
        undistorted_frame = cv2.undistort(
            frame, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # Convert frame to grayscale (required for findChessboardCorners)
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners in the undistorted image
        ret, corners = cv2.findChessboardCorners(gray_frame, pattern_size)

        # If corners are detected, calculate real-world size
        if ret:
            # Refine the corner positions for better accuracy
            corners = cv2.cornerSubPix(
                gray_frame,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )

            # Draw chessboard corners on the frame for visualization
            cv2.drawChessboardCorners(undistorted_frame, pattern_size, corners, ret)

            # Calculate the distance between two adjacent corners in the undistorted image
            pixel_square_size = np.linalg.norm(corners[1] - corners[0])

            # Focal length (in pixel units) from camera matrix
            focal_length = camera_matrix[0, 0]  # Fx from camera matrix

            # Calculate the real-world size using the formula:
            # real_size = (measured_pixel_size / focal_length) * known_square_size
            real_size = (pixel_square_size / focal_length) * square_size_real_world

            # Display the real-world square size
            cv2.putText(
                undistorted_frame,
                f"Real-world square size: {real_size:.2f} cm",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Show the undistorted frame with the calculated square size
        cv2.imshow("Real-time Camera Calibration", undistorted_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage with calibration data and known square size
# Replace these with your calibration results
dist_coeffs = np.array([-0.03338521, 0.70699892, -0.0068209, -0.00709152, -2.47205083])
camera_matrix = np.array([
    [763.99344042, 0.0, 286.91242876],
    [0.0, 764.34274545, 225.39792075],
    [0.0, 0.0, 1.0],
])  # Example camera matrix

square_size_real_world = 2.0  # cm (size of one square on the chessboard in real life)
pattern_size = (9, 6)  # Chessboard pattern size (e.g., 9x6 grid of squares)

# Start the real-time capture and measurement function
capture_and_measure_square(
    camera_matrix, dist_coeffs, square_size_real_world, pattern_size
)
