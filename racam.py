import socket
import cv2
import numpy as np
import struct

# Set the server IP and port
server_ip = '192.168.1.34'  # Replace with your Raspberry Pi's IP
server_port = 8000

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
            break  # Exit if no data is received
        
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

        # Check if we received enough data
        if len(data) != size:
            print(f"Received data size: {len(data)}. Expected: {size}. Exiting.")
            break

        # Decode the frame
        frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode the frame. Exiting.")
            break

        # Display the frame
        cv2.imshow('Live Video Feed', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup: Close the OpenCV window and the socket connection
    cv2.destroyAllWindows()
    sock.close()
    print("Socket closed.")
