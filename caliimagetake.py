import cv2
import numpy as np

# Raspberry Pi IP address updated
raspberry_pi_ip = "http://192.168.1.42:5000/video_feed"

# Capture video from the Raspberry Pi stream
cap = cv2.VideoCapture(raspberry_pi_ip)

# Counter to keep track of saved images
image_counter = 0

while True:
    # Read a frame from the stream
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Display the live video stream
    cv2.imshow("Raspberry Pi Camera Stream", frame)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # If 's' key is pressed, save the current frame as a calibration image
    if key == ord('s'):
        image_filename = f'calibration_images/calibration_image_{image_counter}.jpg'
        cv2.imwrite(image_filename, frame)
        print(f"Saved {image_filename}")
        image_counter += 1

    # If 'q' key is pressed, quit the loop
    if key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
