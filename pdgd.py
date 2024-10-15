import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the face detector and the predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # download from dlib model zoo

# Path to the input video file
video_path = 'IMG_1002.MOV'  # Replace with your video file path

# Capture video from the file
cap = cv2.VideoCapture(video_path)

# List to store gaze data
gaze_data = []

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    # Create a mask to isolate the eye region
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Extract the region of the eye
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    # Divide the eye into two parts: left and right
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if right_side_white == 0:
        gaze_ratio = 1
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Left and right eye ratios
        left_eye_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)

        # Calculate gaze ratio
        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2
        is_looking_center = 0

        if gaze_ratio < 0.8:
            gaze_direction = "Looking right"
        elif gaze_ratio > 1.5:
            gaze_direction = "Looking left"
        else:
            gaze_direction = "Looking center"
            is_looking_center = 1

        # Append data to gaze_data list
        gaze_data.append({
            'frame': len(gaze_data),
            'gaze_ratio': gaze_ratio,
            'is_looking_center': is_looking_center
        })

        # Display gaze direction on the frame
        cv2.putText(frame, gaze_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key to stop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Convert gaze_data to pandas DataFrame
gaze_df = pd.DataFrame(gaze_data)

# Export to CSV
gaze_df.to_csv('gaze_data.csv', index=False)
print('Gaze data saved to gaze_data.csv')

# Plot the results
# Scatter plot for gaze ratio and center look detection
plt.scatter(gaze_df['frame'], gaze_df['gaze_ratio'], c=gaze_df['is_looking_center'], cmap='coolwarm', label='Gaze Ratio')
plt.title('Gaze Ratio Over Time')
plt.xlabel('Frame')
plt.ylabel('Gaze Ratio')
plt.colorbar(label='Is Looking Center (1 = True, 0 = False)')
plt.legend()
plt.show()
