import cv2
import numpy as np
import glob

# Configuration
chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
square_size = 25  # Square size in mm
image_path = 'cali/*.png'  # Update with the path to your images
test_image_path = 'cali/1.png'  # Update with the path to your test image

# Prepare object points based on the chessboard size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to hold object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load chessboard images
images = glob.glob(image_path)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Optional: Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration result
np.savez('calibration_data.npz', mtx=mtx, dist=dist)

# Create the calibration matrix using the intrinsic parameters
calibration_matrix = np.array(mtx)

# Print the calibration matrix and distortion coefficients
print("Camera Calibration Matrix:")
print(calibration_matrix)
print("\nDistortion Coefficients:")
print(dist)

# Load the saved calibration data
calib_data = np.load('calibration_data.npz')
mtx = calib_data['mtx']
dist = calib_data['dist']

# Load a test image to undistort
img = cv2.imread(test_image_path)
h, w = img.shape[:2]

# New camera matrix
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv2.undistort(img, mtx, dist, None, new_mtx)

# Crop the image (optional)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# Show the original and undistorted images
cv2.imshow('Original', img)
cv2.imshow('Undistorted', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
