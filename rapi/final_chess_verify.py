import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls

# Load the saved calibration data from the npz file
calibration_data = np.load("camera_calibration_data.npz")
camera_matrix = calibration_data["camera_matrix"]
dist_coeffs = calibration_data["dist_coeffs"]

# Initialize camera
picam2 = Picamera2()
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
    "AnalogueGain": 1.0,
})

# Chessboard dimensions (inner corners per row and column)
chessboard_size = (9, 6)  # 9x6 means 8x5 squares
square_size = 25  # size of each square in mm

# Start camera
picam2.start()

# Capture a frame for measuring chessboard square size
print("Press 's' to start measurement, 'q' to quit.")

while True:
    # Capture frame in RGBA format
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Undistort the captured image using the calibration data
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted_image = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # Convert to grayscale for corner detection
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine corner positions
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        # Draw and display the corners
        cv2.drawChessboardCorners(undistorted_image, chessboard_size, corners, ret)
        #####
        # Compute the distance between two corners
        point1 = corners[0][0]  # First corner
        point2 = corners[1][0]  # Adjacent corner in the same row

        # Calculate the pixel distance between the two points
        pixel_distance = np.linalg.norm(point1 - point2)

        # Calculate the real-world distance using the known size of a square
        real_world_distance = square_size  # 25mm is the size of each square

        # Calculate pixel-to-mm ratio
        pixel_to_mm_ratio = real_world_distance / pixel_distance
        #####
        # Overlay the measurements on the image
        cv2.putText(
            undistorted_image,
            f"Pixel distance: {pixel_distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            undistorted_image,
            f"Real-world distance: {real_world_distance:.2f} mm",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            undistorted_image,
            f"Pixel-to-mm ratio: {pixel_to_mm_ratio:.4f} mm/px",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    else:
        # Display message when chessboard is not detected
        cv2.putText(
            undistorted_image,
            "Chessboard Not Detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Show the live preview with overlay
    cv2.imshow("Live Measurement Preview", undistorted_image)

    # Wait for key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()
