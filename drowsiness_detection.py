
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import winsound
import threading
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# PARAMETERS
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

counter: int = 0
alarm_on: bool = False
# MEDIAPIPE FACE MESH SETUP
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
if not os.path.exists(model_path):
    model_path = 'face_landmarker.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# FUNCTION: PLAY ALARM (WINDOWS)

def sound_alarm():
    while alarm_on:
        winsound.Beep(2500, 1000)


# FUNCTION: EYE ASPECT RATIO
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)
# START VIDEO CAPTURE

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible")
    exit()

print("[INFO] MediaPipe Drowsiness Detection Started... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        h, w, _ = frame.shape

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:

                left_eye_points = []
                right_eye_points = []

                for idx in LEFT_EYE:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    left_eye_points.append((x, y))

                for idx in RIGHT_EYE:
                    x = int(face_landmarks[idx].x * w)
                    y = int(face_landmarks[idx].y * h)
                    right_eye_points.append((x, y))

                leftEAR = eye_aspect_ratio(left_eye_points)
                rightEAR = eye_aspect_ratio(right_eye_points)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw eye contours
                cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], True, (0, 255, 0), 1)

                # Drowsiness logic
                if ear < EAR_THRESHOLD:
                    counter += 1  # type: ignore


                    if counter >= CONSEC_FRAMES:
                        if not alarm_on:
                            alarm_on = True
                            threading.Thread(target=sound_alarm, daemon=True).start()

                        cv2.putText(frame, "DROWSINESS ALERT!",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2)
                else:
                    counter = 0
                    alarm_on = False

                cv2.putText(frame, f"EAR: {ear:.2f}",
                            (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2)

        else:
            counter = 0
            alarm_on = False

        cv2.imshow("MediaPipe Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System Stopped Successfully.")
