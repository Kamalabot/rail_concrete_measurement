from picamera2 import Picamera2
import cv2
from libcamera import controls

# Initialize the camera
picam2 = Picamera2()

# Define the desired width and height
width = 1920  # example width
height = 1080  # example height

# Create a still configuration with the default format and custom size
still_config = picam2.create_still_configuration(main={"size": (width, height)})

# Configure the camera for the still capture
picam2.configure(still_config)

# Set manual lens position to 1.70 (fixed focus)
picam2.set_controls({"LensPosition": 1.70})

# Enable noise reduction and set a lower gain to reduce noise
picam2.set_controls({
    "AeEnable": True,                 # Enable auto exposure
    "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,  # High-quality noise reduction
    "AnalogueGain": 1.0               # Lower gain to reduce noise (1.0 is often the minimum)
})

# Start the camera
picam2.start()

# Live preview with OpenCV
while True:
    # Capture preview frame, assumed in RGBA format
    frame = picam2.capture_array()

    # Convert from RGBA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Overlay the fixed lens position on the frame
    text = f"Lens Position: 1.70"
    cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show preview window using OpenCV
    cv2.imshow("Live Preview", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close OpenCV window
picam2.stop()
cv2.destroyAllWindows()
