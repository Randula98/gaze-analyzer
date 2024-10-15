import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import for plotting

# Load face and eye detectors from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this path is correct

def detect_eyes(landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], np.int32)
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], np.int32)

    left_eye_bbox = cv2.boundingRect(left_eye)
    right_eye_bbox = cv2.boundingRect(right_eye)

    return left_eye_bbox, right_eye_bbox, left_eye, right_eye

def draw_eye_boxes(frame, left_eye_bbox, right_eye_bbox):
    cv2.rectangle(frame, (left_eye_bbox[0], left_eye_bbox[1]), 
                  (left_eye_bbox[0] + left_eye_bbox[2], left_eye_bbox[1] + left_eye_bbox[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_eye_bbox[0], right_eye_bbox[1]), 
                  (right_eye_bbox[0] + right_eye_bbox[2], right_eye_bbox[1] + right_eye_bbox[3]), (0, 255, 0), 2)

def detect_gaze_direction(left_eye_bbox, right_eye_bbox, left_eye, right_eye):
    left_eye_center_y = (left_eye[1][1] + left_eye[4][1]) // 2
    right_eye_center_y = (right_eye[1][1] + right_eye[4][1]) // 2

    left_eye_bbox_center_y = left_eye_bbox[1] + left_eye_bbox[3] // 2
    right_eye_bbox_center_y = right_eye_bbox[1] + right_eye_bbox[3] // 2

    if left_eye_center_y < left_eye_bbox_center_y - 5 and right_eye_center_y < right_eye_bbox_center_y - 5:
        return "Looking Up"
    elif left_eye_center_y > left_eye_bbox_center_y + 5 and right_eye_center_y > right_eye_bbox_center_y + 5:
        return "Looking Down"
    else:
        left_eye_center_x = left_eye_bbox[0] + left_eye_bbox[2] // 2
        right_eye_center_x = right_eye_bbox[0] + right_eye_bbox[2] // 2
        
        if left_eye_center_x < left_eye_bbox[0] - 10:
            return "Looking Left"
        elif right_eye_center_x > right_eye_bbox[0] + right_eye_bbox[2] + 10:
            return "Looking Right"
        else:
            return "Looking Center"

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    gaze_records = []  # List to store gaze records with timestamps

    frame_count = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye_bbox, right_eye_bbox, left_eye, right_eye = detect_eyes(landmarks)
            draw_eye_boxes(frame, left_eye_bbox, right_eye_bbox)

            gaze_direction = detect_gaze_direction(left_eye_bbox, right_eye_bbox, left_eye, right_eye)

            # Determine if looking center
            is_looking_center = gaze_direction == "Looking Center"
            
            # Calculate time in seconds and round to 1 decimal place
            timestamp = round(frame_count / cap.get(cv2.CAP_PROP_FPS), 1)

            # Append timestamp and is_looking_center status to records
            gaze_records.append((timestamp, is_looking_center))
            
            # Print gaze status (True/False) for each frame
            # print(f"Time: {timestamp:.1f}s, Looking Center: {is_looking_center}")

            cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gaze Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame counter

    cap.release()
    cv2.destroyAllWindows()

    # Save gaze records to a DataFrame
    gaze_df = pd.DataFrame(gaze_records, columns=['time', 'is looking'])
    
    # Convert booleans to lowercase strings ('true'/'false')
    gaze_df['is looking'] = gaze_df['is looking'].apply(lambda x: 'true' if x else 'false')
    
    # Export the DataFrame to CSV
    gaze_df.to_csv('gaze_records.csv', index=False)
    
    # Scatter plot showing whether the person is looking at the center over time
    times = gaze_df['time']
    is_looking_center = gaze_df['is looking'].apply(lambda x: 1 if x == 'true' else 0)

    plt.figure(figsize=(10, 6))
    plt.scatter(times, is_looking_center, color='blue', label='Looking Center', alpha=0.6)
    plt.yticks([0, 1], ['Not Looking Center', 'Looking Center'])
    plt.xlabel('Time (s)')
    plt.ylabel('Gaze Direction')
    plt.title('Gaze Detection Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    video_path = 'IMG_1002.MOV'  # Specify the path to your video file
    main(video_path)
