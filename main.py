import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------- Angle Calculation Function --------
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# -------- Pose Connections (Manual Fix) --------
connections = [
    (11,13),(13,15),
    (12,14),(14,16),
    (11,12),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),
    (24,26),(26,28)
]


# -------- Load Pose Model --------
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)


# -------- Squat Counter Variables --------
counter = 0
stage = None


# -------- Start Webcam --------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # -------- Draw Landmarks --------
        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 5, (0,255,0), -1)


        # -------- Draw Skeleton --------
        for connection in connections:

            start = landmarks[connection[0]]
            end = landmarks[connection[1]]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)


        # -------- Elbow Angle --------
        shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        elbow = [landmarks[13].x * w, landmarks[13].y * h]
        wrist = [landmarks[15].x * w, landmarks[15].y * h]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        cv2.putText(frame,
                    str(int(elbow_angle)),
                    tuple(np.array(elbow, dtype=int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)


        # -------- Knee Angle (Squat Detection) --------
        hip = [landmarks[23].x * w, landmarks[23].y * h]
        knee = [landmarks[25].x * w, landmarks[25].y * h]
        ankle = [landmarks[27].x * w, landmarks[27].y * h]

        knee_angle = calculate_angle(hip, knee, ankle)


        # -------- Squat Logic --------
        if knee_angle < 90:
            stage = "down"

        if knee_angle > 160 and stage == "down":
            stage = "up"
            counter += 1


        # -------- Display Counter --------
        cv2.putText(frame,
                    "Squats: " + str(counter),
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)


    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
