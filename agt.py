import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

# Load the face detector and the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib model zoo

# Set the video path (replace 'video_path' with your actual video file path or 0 for webcam)
video_path = 'IMG_1002.MOV'  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Function to extract eye region points
def get_eye_points(facial_landmarks, eye_points):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    return eye_region

# Function to calculate the midpoint of the eye for gaze detection
def get_eye_center(eye_points):
    return np.mean(eye_points, axis=0).astype(np.int32)

# Function to calculate pupil position based on landmarks
def get_gaze_direction(eye_region):
    # Find the centroid (approximate pupil position based on eye landmarks)
    x_coords = eye_region[:, 0]
    eye_width = np.max(x_coords) - np.min(x_coords)
    centroid_x = np.mean(x_coords)

    # Simple threshold to classify gaze direction based on horizontal eye movement
    if centroid_x < np.mean(x_coords) - 0.2 * eye_width:
        return "Looking left"
    elif centroid_x > np.mean(x_coords) + 0.2 * eye_width:
        return "Looking right"
    else:
        return "Looking center"

# List to store gaze data
gaze_data = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)

        # Get the eye points
        left_eye_region = get_eye_points(landmarks, [36, 37, 38, 39, 40, 41])
        right_eye_region = get_eye_points(landmarks, [42, 43, 44, 45, 46, 47])

        # Determine the gaze direction
        left_gaze_direction = get_gaze_direction(left_eye_region)
        right_gaze_direction = get_gaze_direction(right_eye_region)

        gaze_direction = left_gaze_direction if left_gaze_direction == right_gaze_direction else "Looking center"

        # Check if the person is looking center
        is_looking_center = 1 if gaze_direction == "Looking center" else 0

        # Append gaze data to list in the required format
        gaze_data.append({
            'time': frame_count,
            'is_looking_center': is_looking_center
        })
    
    # Show the frame with detected gaze direction
    cv2.imshow("Affective Gaze Tracking", frame)
    frame_count += 1

    # Break on 'ESC' key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert gaze data to pandas DataFrame
gaze_df = pd.DataFrame(gaze_data)

# Export to CSV with the specified columns (time, islookingcenter)
gaze_df.columns = ['time', 'islookingcenter']
gaze_df['islookingcenter'] = gaze_df['islookingcenter'].apply(lambda x: 'true' if x == 1 else 'false')

gaze_df.to_csv('gaze_tracking_data.csv', index=False)
print('Gaze data saved to gaze_tracking_data.csv')

# Plot the gaze direction over time
plt.scatter(gaze_df['time'], gaze_df['islookingcenter'].apply(lambda x: 1 if x == 'true' else 0), c='g', label='Looking Center', alpha=0.6)
plt.title('Gaze Center Detection Over Time')
plt.xlabel('Time (Frame)')
plt.ylabel('Looking Center (1 = True, 0 = False)')
plt.legend()
plt.show()
