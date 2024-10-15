import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls

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
    "AnalogueGain": 1.0
})

# Start camera
picam2.start()

# Chessboard dimensions (number of inner corners per a chessboard row and column)
chessboard_size = (9, 6)  # 9x6 chessboard

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

captured_images = 0  # Count of manually captured images

while True:
    # Capture frame in RGBA format
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        text = "Chessboard Detected! Press 's' to save the image."
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        text = "Chessboard Not Detected"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the live preview with OpenCV
    cv2.imshow('Calibration Preview', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s') and ret:
        # Save points if 's' is pressed and corners are detected
        objpoints.append(objp)
        imgpoints.append(corners)
        captured_images += 1
        print(f"Captured {captured_images} images for calibration.")
    elif key & 0xFF == ord('q'):
        # Exit the loop when 'q' is pressed
        break

# Stop the camera and close OpenCV window
picam2.stop()
cv2.destroyAllWindows()

# Proceed to calibration if there are enough points
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Calibrating camera...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Display calibration results
    if ret:
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs}")

        # Save calibration data to an npz file
        np.savez('camera_calibration_data.npz', 
                 camera_matrix=camera_matrix, 
                 dist_coeffs=dist_coeffs, 
                 rvecs=rvecs, 
                 tvecs=tvecs)

        print("Calibration data saved to 'camera_calibration_data.npz'")
    else:
        print("Camera calibration failed.")
else:
    print("No sufficient images were captured for calibration.")
