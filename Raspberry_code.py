from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)
picam2 = Picamera2()

# Configure the camera
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.set_controls({"AfMode": 0 ,"AfTrigger": 0,"LensPosition": 0.1})
picam2.start()

def generate_frames():
    while True:
        frame = picam2.capture_array()
        # Convert the frame to BGR format
        if frame.shape[2] == 4:  # RGBA image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:  # RGB image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Use your Raspberry Pi's IP address
