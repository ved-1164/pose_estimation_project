import cv2

# Direct imports that work with newer mediapipe builds
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks

class PoseEngine:

    def __init__(self):
        # Initialize pose detector
        self.pose = Pose()
    
    def detect_pose(self, frame):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        results = self.pose.process(rgb)

        # Draw skeleton
        if results.pose_landmarks:
            draw_landmarks(
                frame,
                results.pose_landmarks,
                POSE_CONNECTIONS
            )

        return frame, results

    def get_landmarks(self, frame, results):
        h, w, _ = frame.shape
        landmarks = {}

        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks[idx] = (x, y)

        return landmarks
    