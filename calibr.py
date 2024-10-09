import cv2
import numpy as np
import glob

# Chessboard dimensions
chessboard_size = (9, 6)  # Change this if your chessboard is different
square_size_mm = 25       # The size of a square on your chessboard in millimeters

# Termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the real-world coordinates of the chessboard corners
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size_mm

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load the chessboard images for calibration
images = glob.glob('calibration_images/*.jpg')  # Replace with the path to your images

# Initialize gray variable before the loop
gray = None

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)

        # Refine the corner detection for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if gray is not None:
    # Perform camera calibration to get the intrinsic camera matrix (mtx) and distortion coefficients (dist)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration data to a .npz file
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

    # Display the calibration results
    print("Camera matrix (mtx):\n", mtx)
    print("Distortion coefficients (dist):\n", dist)
    print("Reprojection error:\n", ret)
else:
    print("Calibration failed: No valid images were processed.")
