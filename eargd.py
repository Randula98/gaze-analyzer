import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
import matplotlib.pyplot as plt

# Load the face detector and the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib model zoo

# EAR threshold for blinking detection
EAR_THRESHOLD = 0.25  # Adjust based on real-world testing
CONSECUTIVE_FRAMES = 3  # Number of consecutive frames where eye must be below the threshold to count as a blink

# Set the video path (replace 'video_path' with your actual video file path)
video_path = 'IMG_1002.MOV'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # Vertical distance between p2 and p6
    B = dist.euclidean(eye_points[2], eye_points[4])  # Vertical distance between p3 and p5
    C = dist.euclidean(eye_points[0], eye_points[3])  # Horizontal distance between p1 and p4
    ear = (A + B) / (2.0 * C)
    return ear

# Function to extract eye region points
def get_eye_points(facial_landmarks, eye_indices):
    return np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_indices], np.float32)

# List to store blink and EAR data
ear_data = []
frame_count = 0
blink_count = 0
consec_blink_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get eye points for both eyes
        left_eye_points = get_eye_points(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_points = get_eye_points(landmarks, [42, 43, 44, 45, 46, 47])

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)

        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check for blink (if EAR is below threshold)
        if ear < EAR_THRESHOLD:
            consec_blink_frames += 1
        else:
            if consec_blink_frames >= CONSECUTIVE_FRAMES:
                blink_count += 1
            consec_blink_frames = 0

        # Append EAR data for analysis
        ear_data.append({
            'frame': frame_count,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'average_ear': ear,
            'blink_count': blink_count
        })

        # Display EAR and blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the eye regions
        cv2.polylines(frame, [left_eye_points.astype(np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points.astype(np.int32)], True, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow("EAR-based Blink Detection", frame)
    frame_count += 1

    # Break on 'ESC' key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert EAR data to pandas DataFrame
ear_df = pd.DataFrame(ear_data)

# Export to CSV
ear_df.to_csv('ear_blink_data.csv', index=False)
print('Blink data saved to ear_blink_data.csv')

# Plot the results
plt.plot(ear_df['frame'], ear_df['average_ear'], label='Average EAR')
plt.axhline(y=EAR_THRESHOLD, color='r', linestyle='--', label='EAR Threshold')
plt.title('EAR Over Time')
plt.xlabel('Frame')
plt.ylabel('Eye Aspect Ratio (EAR)')
plt.legend()
plt.show()
