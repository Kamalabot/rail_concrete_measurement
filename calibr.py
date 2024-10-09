import cv2
import numpy as np
import os

# Define parameters for the chessboard
chessboard_size = (9, 6)  # (number of inner corners per chessboard row and column)
square_size = 25.0  # size of a square in mm

# Create object points based on the chessboard size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Folder where calibration images are saved
save_folder = 'calibration_images'

# List all saved images
image_files = [f for f in os.listdir(save_folder) if f.endswith('.png')]

# Process each image and find chessboard corners
for img_file in image_files:
    img_path = os.path.join(save_folder, img_file)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # Refine corner positions
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # Append object points and image points for calibration
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"Processed image: {img_file}")
    else:
        print(f"Could not find chessboard in {img_file}")

# Perform camera calibration if sufficient images were processed
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration results
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)
    print("Calibration completed and saved as 'camera_calibration.npz'.")

    # Display calibration results
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
else:
    print("No valid images found for calibration.")
