import cv2
import numpy as np
import os
from gpiozero import Servo, MotionSensor
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Configuration
AUTHORIZED_FOLDER = "authorized_faces"  # Folder containing images of authorized faces
LOCK_GPIO_PIN = 17  # GPIO pin connected to the lock system
MOTION_GPIO_PIN = 4  # GPIO pin connected to the motion sensor

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size=(640, 480))

# Initialize Servo for Lock
lock = Servo(LOCK_GPIO_PIN)

# Initialize Motion Sensor
motion_sensor = MotionSensor(MOTION_GPIO_PIN)

# Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to load and train facial recognition
def load_trained_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for filename in os.listdir(AUTHORIZED_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(AUTHORIZED_FOLDER, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            faces.append(img)
            labels.append(label_id)
            label_map[label_id] = os.path.splitext(filename)[0]
            label_id += 1
    
    recognizer.train(faces, np.array(labels))
    return recognizer, label_map

# Load trained recognizer and labels
recognizer, label_map = load_trained_faces()

# Function to detect and verify face
def detect_and_verify_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        if confidence < 50:  # Adjust confidence threshold as needed
            return label_map[label]
    
    return None

# Function to lock/unlock the door
def lock_door():
    lock.min()  # Adjust for your servo's locking position
    print("Door locked.")

def unlock_door():
    lock.max()  # Adjust for your servo's unlocking position
    print("Door unlocked.")

# Main loop
print("Security system is running...")
try:
    lock_door()
    time.sleep(1)  # Ensure the system initializes with the door locked

    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        
        if motion_sensor.motion_detected:
            print("Motion detected!")
            user = detect_and_verify_face(image)
            
            if user:
                print(f"Authorized user detected: {user}")
                unlock_door()
                time.sleep(10)  # Keep the door unlocked for 10 seconds
                lock_door()
            else:
                print("Unauthorized face detected or no face detected.")
        
        # Clear the stream for the next frame
        raw_capture.truncate(0)

except KeyboardInterrupt:
    print("Shutting down security system.")
    camera.close()
