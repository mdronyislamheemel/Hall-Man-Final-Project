import requests
import cv2  
import os
import time
import face_recognition
import numpy as np

# Constants
BASE_OUTPUT_DIR = '/home/rony/Documents/Project/Face_Recognition/saved_faces'
KNOWN_FACES_DIR = '/home/rony/Documents/Project/Face_Recognition/Images'
UNKNOWN_FACES_DIR = os.path.join(BASE_OUTPUT_DIR, 'unknown_faces')

# Create directories if they don't exist
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)
if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

DESIRED_FPS = 15
PERIOD_SECONDS = 10
MATCH_THRESHOLD = 0.7  # Tighter matching threshold
MIN_FACE_WIDTH = 30
MIN_FACE_HEIGHT = 30

# API Constants
GET_API_URL = "http://localhost/api/students"
POST_API_URL = "http://localhost/api/attendance"

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to fetch student data from GET API
def fetch_student_data_from_api():
    try:
        response = requests.get(GET_API_URL)
        if response.status_code == 200:
            students = response.json()['data']
            if not os.path.exists(KNOWN_FACES_DIR):
                os.makedirs(KNOWN_FACES_DIR)

            for student in students:
                image_url = student['image']
                student_id = student['id']
                image_response = requests.get(image_url)
                image_path = os.path.join(KNOWN_FACES_DIR, f'{student_id}.jpg')
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_response.content)

            return students
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

# Load known faces from the directory
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_ids = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                student_id = filename.split('.')[0]  # Get student ID from filename
                known_face_ids.append(student_id)
    
    return known_face_encodings, known_face_ids

# POST attendance data to API
def post_attendance_to_api(student_id):
    payload = {
        "student_id": student_id,
        "hall_id": 1
    }
    response = requests.post(POST_API_URL, json=payload)
    
    if response.status_code == 201:
        print(f"Attendance posted for student {student_id}")
    else:
        print(f"Failed to post attendance for student {student_id}. Status code: {response.status_code}")

# POST unknown face data to API
def post_unknown_face_to_log(image_path, timestamp):
    payload = {
        "student_id": "unknown",  # Mark the student as unknown
        "hall_id": 1,
        "timestamp": timestamp
    }
    response = requests.post(POST_API_URL, json=payload)
    
    if response.status_code == 201:
        print(f"Unknown face logged at {timestamp}")
    else:
        print(f"Failed to log unknown face. Status code: {response.status_code}")

# Filter overlapping faces
def non_overlapping_faces(faces):
    filtered_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        overlap = False
        for j, (x2, y2, w2, h2) in enumerate(faces):
            if i != j:
                if x < x2 + w2 and x + w > x2 and y < y2 + h2 and y + h > y2:
                    overlap = True
                    break
        if not overlap:
            filtered_faces.append((x, y, w, h))
    return filtered_faces

# Main code

# Fetch students and save their images
students_data = fetch_student_data_from_api()

# Load known face encodings
known_face_encodings, known_face_ids = load_known_faces(KNOWN_FACES_DIR)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

frame_count_within_period = 0
time_folder = None
last_posted_time = {}  # To store the last post time for each student
last_encoding = None
last_encoding_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count_within_period += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT))
    faces = non_overlapping_faces(faces)  # Filter overlapping faces

    if len(faces) > 0:
        if time_folder is None or frame_count_within_period >= int(cap.get(cv2.CAP_PROP_FPS) * PERIOD_SECONDS):
            time_folder = os.path.join(BASE_OUTPUT_DIR, time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(time_folder)
            frame_count_within_period = 0

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_locations = [(y, x + w, y + h, x)]
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            if face_encodings:
                face_encoding = face_encodings[0]

                if last_encoding is not None:
                    encoding_distance = face_recognition.face_distance([last_encoding], face_encoding)
                    if encoding_distance < MATCH_THRESHOLD and time.time() - last_encoding_time < PERIOD_SECONDS:
                        continue  # Skip if same face detected too soon

                last_encoding = face_encoding
                last_encoding_time = time.time()

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] < MATCH_THRESHOLD:
                    student_id = known_face_ids[best_match_index]
                    color = (0, 255, 0)  # Green for known faces

                    current_time = time.time()
                    last_time = last_posted_time.get(student_id, 0)

                    if current_time - last_time > PERIOD_SECONDS:
                        post_attendance_to_api(student_id)
                        last_posted_time[student_id] = current_time
                else:
                    # Handle unknown face
                    color = (0, 0, 255)  # Red for unknown faces
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    unknown_face_path = os.path.join(UNKNOWN_FACES_DIR, f'unknown_{timestamp}.jpg')

                    # Save the unknown face image
                    cv2.imwrite(unknown_face_path, face_roi)
                    print(f"Unknown face saved at {unknown_face_path}")

                    # Post "unknown" face to the log
                    post_unknown_face_to_log(unknown_face_path, timestamp)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
