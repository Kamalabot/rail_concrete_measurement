import time
import cv2
import numpy as np
import glob

# Chessboard settings
pattern_size = (9, 6)  # number of internal corners
square_size = 1.0  # size of a square in real-world units
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (like (0,0,0), (1,0,0), ..., (8,5,0))
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
obj_points = []  # 3d points in real-world space
img_points = []  # 2d points in image plane

# Load images
images = glob.glob("../calibration_images/IMG20241006213925/*.jpg")

for image_path in images[:1]:
    print(f"Loading the image {image_path.split('/')[-1]}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)
    find_cbd = time.time()
    print("Time taken to find chess board: {find_cbd - start_time}")
    print(f"Is there a ret: {ret}")
    print(f"Is there corners: {corners}")
    # If found, add object points and image points
    if ret:
        corners_refined = cv2.cornerSubPix(
            gray_image, corners, (11, 11), (-1, -1), criteria
        )
        img_points.append(corners_refined)
        obj_points.append(objp)
    else:
        print(f"Chessboard not detected in image {image_path}")

# Ensure we have valid images
if len(img_points) == 0:
    print("Error: No valid images for calibration.")
else:
    start_cal = time.time()
    print("Starting to caliberate")
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray_image.shape[::-1], None, None
    )
    end_cal = time.time()
    print(f"Time taken for caliberation: {end_cal - start_cal}")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients:\n{dist}")
