# AI Pose Detection and Squat Counter

This project implements a real-time human pose detection and squat counter using Python and computer vision.

The system detects human body landmarks using a webcam, draws a skeleton on the detected body, calculates joint angles, and counts squats automatically.

## Technologies Used

- Python
- MediaPipe
- OpenCV
- NumPy

## Features

- Real-time webcam pose detection
- Detection of 33 body landmarks
- Skeleton visualization
- Joint angle calculation
- Automatic squat counter

## Project Structure

```
pose_estimation_project
│
├── main.py
├── pose_landmarker_lite.task
└── README.md
```

## Installation

Install the required libraries:

```
pip install opencv-python mediapipe numpy
```

## Run the Project

Navigate to the project folder and run:

```
python main.py
```

Press **Q** to exit the webcam window.

## How It Works

1. The webcam captures video frames.
2. MediaPipe detects body landmarks in each frame.
3. Landmarks are connected to form a skeleton.
4. Joint angles are calculated using three body points.
5. The knee angle is used to detect squat motion.
6. When a full squat movement is detected, the counter increases.

## Squat Detection Logic

Three body landmarks are used:

Hip → Knee → Ankle

Rules used:

- Knee angle < 90° → Squat Down
- Knee angle > 160° → Standing Up

When the system detects a Down → Up movement, the squat counter increases.

## Applications

- Fitness tracking
- Exercise monitoring
- Rehabilitation analysis
- Posture detection

## Author

Computer Vision Mini Project using Python.
