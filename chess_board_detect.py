import cv2
import numpy as np

# Load the image of the chessboard
image = cv2.imread("chessboard.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the number of inner corners along the rows and columns of the chessboard
# For example, for a 9x6 chessboard, the inner corners are 8x5 (one less than the squares)
chessboard_size = (8, 5)

# Find the chessboard corners in the grayscale image
ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

if ret:
    # Draw the detected corners on the original image
    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

    # Optional: Refine corner locations for sub-pixel accuracy
    corners_refined = cv2.cornerSubPix(
        gray_image,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    # Show the image with detected corners
    cv2.imshow("Chessboard Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # You can now use the detected corners (corners_refined) for camera calibration
    print("Detected corners:", corners_refined)
else:
    print("Chessboard corners not found.")
