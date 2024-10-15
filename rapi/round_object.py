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

# Set manual controls for exposure and gain
picam2.set_controls({
    "LensPosition": 1.70,  # Manual focus
    "AeEnable": True,  # Auto exposure disabled for manual control
    "ExposureTime": 8500,  # Set shutter speed to 8500 μs (8.5 ms)
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
    "AnalogueGain": 1.0,  # Set low gain for reduced noise
})

# Known pixel-to-mm ratio (from your calibration data)
pixel_to_mm_ratio = (
    0.228  # Example: 0.228 mm per pixel, adjust based on your calibration
)

# Start camera
picam2.start()

print("Press 'q' to quit.")

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

    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

    # Apply a slight blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # Detect circles using Hough Circle Transform with reduced sensitivity
    # For the object that we want to detect, and measure we need to change the
    # below Class.
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        # This two params can be tweaked
        param1=150,  # Higher value makes edge detection less sensitive
        param2=50,  # Higher value makes circle detection more selective
        # decide which size of the circles to capture, like a size filter
        minRadius=30,  # Set a minimum radius to avoid detecting small circles
        maxRadius=150,
    )  # Set a maximum radius for large circles

    if circles is not None:
        circles = np.uint16(
            np.around(circles)
        )  # Round the circle coordinates and radii
        for circle in circles[0, :]:
            # Get circle center and radius
            center_x, center_y, radius_px = circle[0], circle[1], circle[2]

            # Draw the circle in the image
            cv2.circle(
                undistorted_image, (center_x, center_y), radius_px, (0, 255, 0), 2
            )
            cv2.circle(
                undistorted_image, (center_x, center_y), 2, (0, 0, 255), 3
            )  # Mark the center

            # Convert the radius from pixels to mm using the pixel-to-mm ratio
            radius_mm = radius_px * pixel_to_mm_ratio

            # Display the circle's radius in pixels and real-world size in mm
            cv2.putText(
                undistorted_image,
                f"Radius: {radius_mm:.2f}mm",
                (center_x - 50, center_y - radius_px - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    else:
        cv2.putText(
            undistorted_image,
            "No circles detected",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    # Show the live preview with measurement overlay
    cv2.imshow("Circle Measurement", undistorted_image)

    # Wait for key press
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()
