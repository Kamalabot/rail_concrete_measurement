import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls

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
picam2.set_controls({
    "LensPosition": 1.70,
    "AeEnable": True,
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
    "AnalogueGain": 1.0
})

# Start camera
picam2.start()

# Capture a frame for distortion check
print("Press 's' to capture an image and 'q' to quit.")

while True:
    # Capture frame in RGBA format
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Show the live preview
    cv2.imshow('Live Preview', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'):
        # Capture image and undistort it
        print("Captured image for distortion check.")
        original_image = frame.copy()
        
        # Undistort the captured image
        h, w = original_image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(original_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Crop the undistorted image to remove black borders
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        # Resize the undistorted image to match the original size
        undistorted_image_resized = cv2.resize(undistorted_image, (original_image.shape[1], original_image.shape[0]))

        # Show the original and undistorted images side by side
        combined_image = np.hstack((original_image, undistorted_image_resized))
        cv2.imshow('Original (Left) vs Undistorted (Right)', combined_image)

        print("Press 'q' to quit or 's' to capture again.")

    elif key & 0xFF == ord('q'):
        # Exit the loop
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()
