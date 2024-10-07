import cv2
import numpy as np
import time

# Chessboard settings
pattern_size = (
    9,
    6,
)  # Define the number of inner corners per a chessboard row and column
square_size = 1.0  # Size of a square on your printed chessboard in your chosen unit (e.g., centimeters)

# Define criteria for corner sub-pixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = (
    np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size
)

# Arrays to store object points and image points from all frames
obj_points = []  # 3d points in real-world space
img_points = []  # 2d points in image plane

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Number of images to capture for calibration
num_images = 20
captured_images = 0

while captured_images < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, add them to the list and refine the corner positions
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_refined)
        obj_points.append(objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, pattern_size, corners_refined, ret)
        captured_images += 1
        print(f"Captured {captured_images}/{num_images} images for calibration.")

    # Display the frame with detected chessboard corners
    cv2.imshow("Calibration", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Pause for a moment so user can adjust the board
    time.sleep(0.5)

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Ensure enough images were captured for calibration
if len(img_points) > 0:
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    if ret:
        print("Camera calibrated successfully!")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")
    else:
        print("Calibration failed.")
else:
    print("No valid images captured for calibration.")
