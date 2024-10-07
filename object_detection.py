import socket
import cv2
import numpy as np
import struct

# Load the calibration data (camera matrix and distortion coefficients)
with np.load('calibration_data.npz') as data:
    camera_matrix = data['mtx']
    dist_coeffs = data['dist']

# Load the MobileNet SSD model (Caffe format)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define class labels for detection (21 classes for PASCAL VOC)
class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

# Server setup for receiving frames from Raspberry Pi
server_ip = '192.168.1.34'  # Replace with your Raspberry Pi IP address
server_port = 8000

def undistort_image(frame):
    """Undistort the input frame using the calibration data."""
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_frame

def measure_object_dimensions(box, frame_shape):
    """Estimate the real-world dimensions of detected objects."""
    # Assuming fixed distance and using a known focal length
    h, w = frame_shape[:2]
    
    # Use camera calibration and the bounding box to estimate object size
    # This will need to be refined based on the actual real-world object size and distance from the camera
    (startX, startY, endX, endY) = box.astype("int")
    width_pixels = endX - startX
    height_pixels = endY - startY
    
    # Example: Estimating real size assuming focal length and known object distance
    focal_length = camera_matrix[0, 0]  # Use fx from the camera matrix
    real_object_width = 0.1  # Assume known object width (e.g., 10 cm) for the measurement
    
    distance = (real_object_width * focal_length) / width_pixels
    object_real_height = (height_pixels * distance) / focal_length
    
    return distance, object_real_height

try:
    # Create a socket connection to the server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_ip, server_port))
    print("Connected to server.")

    while True:
        # Receive the size of the frame
        size_data = sock.recv(4)
        if not size_data:
            print("No data received. Exiting.")
            break

        # Get the size of the incoming frame
        size = struct.unpack('>I', size_data)[0]

        # Receive the actual frame data
        data = b''
        while len(data) < size:
            packet = sock.recv(4096)
            if not packet:
                print("Incomplete data received. Exiting.")
                break
            data += packet

        # Check if the received data size matches the expected size
        received_size = len(data)
        if received_size != size:
            print(f"Received data size: {received_size}. Expected: {size}. Size mismatch. Exiting.")
            break

        # Decode the frame
        frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode the frame. Exiting.")
            break

        # Undistort the frame using the camera calibration data
        undistorted_frame = undistort_image(frame)

        # Prepare the frame for the MobileNet SSD model
        blob = cv2.dnn.blobFromImage(undistorted_frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        # Perform object detection
        detections = net.forward()

        # Loop over the detections
        h, w = undistorted_frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.rectangle(undistorted_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(undistorted_frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Estimate and display object dimensions
                distance, object_real_height = measure_object_dimensions(box, undistorted_frame.shape)
                dimension_text = f"Distance: {distance:.2f}m, Height: {object_real_height:.2f}m"
                cv2.putText(undistorted_frame, dimension_text, (startX, startY + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detected objects and measurements
        cv2.imshow('Object Detection with Measurement', undistorted_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cv2.destroyAllWindows()
    sock.close()
    print("Socket closed.")
