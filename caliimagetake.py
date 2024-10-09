import cv2
import os

# Define parameters for the chessboard
chessboard_size = (9, 6)  # (number of inner corners per chessboard row and column)

# Create a directory to save calibration images if it doesn't exist
save_folder = 'calibration_images'
os.makedirs(save_folder, exist_ok=True)

# Set the video stream URL for the Raspberry Pi camera
video_stream_url = 'http://192.168.1.42:5000/video_feed'  # Adjust if necessary

# Start capturing images from the video stream
cap = cv2.VideoCapture(video_stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 's' to save images of the chessboard pattern, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the live frame
    cv2.imshow('Chessboard', frame)

    # Press 's' to save the image and 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = os.path.join(save_folder, f'calibration_image_{len(os.listdir(save_folder)) + 1}.png')
        cv2.imwrite(img_name, frame)  # Save the raw frame without any markings
        print(f"Saved image: {img_name}")
    elif key == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

print(f"Saved images in {save_folder}. You can now proceed with calibration.")
